"""Verify pinned Torchvision artifacts against every tensor retained by our adapters.

Metadata resolution alone is not proof of loaded pretrained parameters. The verified
builder checks the cached artifact hash, reads it independently with weights-only
CPU loading, constructs the existing adapter, and compares retained tensor values.
"""

from collections.abc import Mapping
import hashlib
from pathlib import Path
import pickle
import re
from urllib.parse import urlparse

import torch
from torch import Tensor, nn
import torchvision
from torchvision import models

from src.models.pretrained import ModelConfig, PretrainedClassifier, create_pretrained_model


def resolve_weight_info(config: ModelConfig) -> dict:
    """Resolve the existing config's pinned enum without downloading or reading cache."""
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    enum_name, member = config.weights.split(".")
    weight = getattr(getattr(models, enum_name), member)
    url = weight.url
    filename = Path(urlparse(url).path).name
    match = re.search(r"-([a-f0-9]+)\.", filename)
    if match is None or len(match.group(1)) < 8:
        raise ValueError("official weight filename has no supported SHA-256 hash prefix")
    return {
        "architecture": config.architecture,
        "weight_id": config.weights,
        "url": url,
        "filename": filename,
        "hash_prefix": match.group(1),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
    }


def prepare_official_weight(config, *, offline=False, progress=True) -> dict:
    """Reuse or download one Torch Hub checkpoint and verify its filename hash.

    Existing corrupt files fail rather than being replaced silently. Set
    ``torch.hub.set_dir`` before calling to use an explicitly staged cache.
    """
    if type(offline) is not bool or type(progress) is not bool:
        raise TypeError("offline and progress must be bools")
    info = resolve_weight_info(config)
    path = Path(torch.hub.get_dir()) / "checkpoints" / info["filename"]
    if not path.exists():
        if offline:
            raise FileNotFoundError(f"official weight checkpoint missing in offline mode: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            info["url"], str(path), hash_prefix=info["hash_prefix"], progress=progress,
        )
    if not path.is_file():
        raise ValueError(f"official weight checkpoint is not a file: {path}")
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size_bytes += len(block)
    sha256 = digest.hexdigest()
    if not sha256.startswith(info["hash_prefix"]):
        raise ValueError(f"official checkpoint SHA-256 mismatch: {path}")
    return info | {
        "cache_path": str(path), "sha256": sha256,
        "size_bytes": size_bytes, "hash_verified": True,
    }


def _native_key(architecture: str, adapter_key: str) -> str:
    if adapter_key.startswith("encoder."):
        suffix = adapter_key.removeprefix("encoder.")
        return suffix if architecture == "regnet_y_3_2gf" else "features." + suffix
    if architecture == "convnext_tiny" and adapter_key.startswith("encoder_norm."):
        return "classifier.0." + adapter_key.removeprefix("encoder_norm.")
    raise ValueError(f"unexpected retained adapter state: {adapter_key}")


def _verify_head(model: PretrainedClassifier) -> dict:
    if (not isinstance(model.head, nn.Sequential) or len(model.head) != 2
            or not isinstance(model.head[1], nn.Linear)
            or model.head[1].out_features != 200
            or model.head[1].in_features != model.spec.encoder_channels
            or model.config.num_classes != 200):
        raise ValueError("replacement head must be a 200-output linear classification head")
    if not all(parameter.requires_grad for parameter in model.head.parameters()):
        raise ValueError("all replacement head parameters must be trainable")
    return {"out_features": 200, "trainable": True}


def verify_backbone_state(model, state_dict) -> dict:
    """Compare exact retained parameter/buffer keys, shapes, dtypes and values.

    Only the two native 1000-class head tensors are excluded from value comparison;
    their exact keys, types and native dimensions must still be present.
    """
    if not isinstance(model, PretrainedClassifier):
        raise TypeError("model must be a PretrainedClassifier")
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint state_dict must be a tensor mapping")
    _verify_head(model)
    architecture = model.config.architecture
    head_prefix = {"convnext_tiny": "classifier.2", "efficientnet_b3": "classifier.1", "regnet_y_3_2gf": "fc"}[architecture]
    head_shapes = {
        head_prefix + ".weight": (1000, model.spec.encoder_channels),
        head_prefix + ".bias": (1000,),
    }
    retained = {
        _native_key(architecture, key): value
        for key, value in model.state_dict().items() if not key.startswith("head.")
    }
    expected = set(retained) | set(head_shapes)
    actual = set(state_dict)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise ValueError(
            f"checkpoint state keys differ; missing={sorted(missing)!r}, "
            f"unexpected={sorted(unexpected, key=str)!r}"
        )
    for key, shape in head_shapes.items():
        value = state_dict[key]
        if not isinstance(value, Tensor):
            raise TypeError(f"checkpoint value must be a tensor: {key}")
        if tuple(value.shape) != shape:
            raise ValueError(f"native 1000-class head shape mismatch: {key}")
    for key, actual_tensor in retained.items():
        expected_tensor = state_dict[key]
        if not isinstance(expected_tensor, Tensor):
            raise TypeError(f"checkpoint value must be a tensor: {key}")
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(f"retained tensor shape mismatch: {key}")
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(f"retained tensor dtype mismatch: {key}")
        if not torch.equal(actual_tensor.detach().cpu(), expected_tensor.detach().cpu()):
            raise ValueError(f"retained tensor value mismatch: {key}")
    return {
        "status": "verified",
        "retained_tensor_count": len(retained),
        "parameter_tensor_count": sum(not key.startswith("head.") for key, _ in model.named_parameters()),
        "buffer_tensor_count": sum(not key.startswith("head.") for key, _ in model.named_buffers()),
        "excluded_native_head_keys": sorted(head_shapes),
    }


def build_verified_pretrained_model(config, *, offline=False, progress=True):
    """Construct the existing pretrained adapter and return independent artifact proof.

    A loading or comparison failure propagates; random construction is never used.
    No forward/backward or device-execution claim is made by this weight proof.
    """
    artifact = prepare_official_weight(config, offline=offline, progress=progress)
    try:
        state_dict = torch.load(artifact["cache_path"], map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, pickle.UnpicklingError, ValueError, TypeError) as exc:
        raise ValueError("official checkpoint could not be safely loaded as weights-only CPU state") from exc
    if not isinstance(state_dict, Mapping):
        raise TypeError("official checkpoint state_dict must be a tensor mapping")
    model = create_pretrained_model(config, load_weights=True, progress=progress)
    backbone = verify_backbone_state(model, state_dict)
    return model, artifact | {
        "status": "verified", "backbone": backbone, "head": _verify_head(model),
    }
