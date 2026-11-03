"""CNN encoders with explicit weights and a separate CUB-200 head.

Construction loads the pinned ImageNet weights by default. Pass
``load_weights=False`` explicitly for offline construction with random parameters.
Native preprocessing is metadata; it does not change the project's data transforms.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision import models

from .finetuning import (
    apply_parameter_policy,
    apply_training_policy,
    validate_fine_tuning,
)


_WEIGHT_IDS = {
    "convnext_tiny": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
    "efficientnet_b3": "EfficientNet_B3_Weights.IMAGENET1K_V1",
    "regnet_y_3_2gf": "RegNet_Y_3_2GF_Weights.IMAGENET1K_V2",
    "convnextv2_base": "timm/convnextv2_base.fcmae_ft_in22k_in1k_384",
}
_ENCODER_CHANNELS = {"convnext_tiny": 768, "efficientnet_b3": 1536, "regnet_y_3_2gf": 1512,
                     "convnextv2_base": 1024}
_PARTIAL_PATHS = {
    "convnext_tiny": ("6", "7"),
    "efficientnet_b3": ("7", "8"),
    "regnet_y_3_2gf": ("trunk_output.block4",),
    "convnextv2_base": ("stages.3",),
}

# Paper-author ImageNet-22K -> ImageNet-1K weights, mirrored in timm's pinned repository.
FB22K_WEIGHT_ID = "facebook/convnext_tiny.in22k_ft_in1k_384"
FB22K_WEIGHT_INFO = {
    "url": "https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384/resolve/32c741ba106f9e062174ef829354a16719c3dcac/pytorch_model.bin",
    "revision": "32c741ba106f9e062174ef829354a16719c3dcac",
    "hash_prefix": "c1253ad2d4a751cb0b1fb92a850f99e8f426a701d6e7600b03a490557d77bdcd",
    "filename": "convnext_tiny-fb22k-c1253ad2d4a751cb0b1fb92a850f99e8f426a701d6e7600b03a490557d77bdcd.pth",
    "source_profile": "convnext_tiny.fb_in22k_ft_in1k_384",
}
V2BASE_WEIGHT_ID = _WEIGHT_IDS["convnextv2_base"]
V2BASE_WEIGHT_INFO = {
    "url": "https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384/resolve/bea9dc139d5981aff6ce4d6d9690086e45ad8b56/model.safetensors",
    "revision": "bea9dc139d5981aff6ce4d6d9690086e45ad8b56",
    "hash_prefix": "972bcbb8d9965eca7a2f72d9592f54dfc890f9e6aef920cea4269963f46f2f42",
    "filename": "convnextv2_base.fcmae_ft_in22k_in1k_384-972bcbb8d9965eca7a2f72d9592f54dfc890f9e6aef920cea4269963f46f2f42.safetensors",
    "source_profile": "convnextv2_base.fcmae_ft_in22k_in1k_384",
}


def _weight_id(architecture: str, weights: str | None = None) -> str:
    if not isinstance(architecture, str) or architecture not in _WEIGHT_IDS:
        raise ValueError(f"architecture must be one of {tuple(_WEIGHT_IDS)}")
    allowed = (_WEIGHT_IDS[architecture],)
    if architecture == "convnext_tiny":
        allowed += (FB22K_WEIGHT_ID,)
    if weights is not None and (not isinstance(weights, str) or weights not in allowed):
        raise ValueError(f"weights for {architecture} must be one of {allowed!r} or None")
    return _WEIGHT_IDS[architecture] if weights is None else weights


def _weight_enum(architecture: str):
    enum_name, member = _weight_id(architecture).split(".")
    return getattr(getattr(models, enum_name), member)


@dataclass(frozen=True)
class ModelConfig:
    """Validated immutable construction settings; ``None`` selects the pinned weight ID."""

    architecture: str
    weights: str | None = None
    num_classes: int = 200
    fine_tuning: str = "frozen"
    head_dropout: float = 0.0
    freeze_bn_stats: bool = True

    def __post_init__(self) -> None:
        pinned = _weight_id(self.architecture, self.weights)
        object.__setattr__(self, "weights", pinned)
        if type(self.num_classes) is not int or self.num_classes != 200:
            raise ValueError("num_classes must be the integer 200")
        validate_fine_tuning(self.fine_tuning, self.freeze_bn_stats)
        if (type(self.head_dropout) not in (int, float)
                or not math.isfinite(self.head_dropout)
                or not 0 <= self.head_dropout < 1):
            raise ValueError("head_dropout must be a finite number in [0, 1)")


@dataclass(frozen=True)
class ModelSpec:
    """Serializable architecture metadata, independent of any loaded parameters.

    ``encoder_channels`` is the spatial/pooled channel count; ``output_stride``
    is 32 for these encoders. ``gradcam_target_path`` resolves on the wrapper.
    The native recipe describes the selected weight preset, not baseline input.
    """

    architecture: str
    weight_id: str
    native_preprocessing: dict
    encoder_channels: int
    output_stride: int
    gradcam_target_path: str


def get_model_spec(architecture: str, weights: str | None = None) -> ModelSpec:
    """Resolve the pinned weight recipe locally, without model construction or I/O."""
    weight_id = _weight_id(architecture, weights)
    if weight_id in (FB22K_WEIGHT_ID, V2BASE_WEIGHT_ID):
        native = {
            "resize_size": [384], "crop_size": [384], "interpolation": "bicubic",
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
            "antialias": True, "crop_mode": "squash", "crop_pct": 1.0,
        }
    else:
        transforms = _weight_enum(architecture).transforms()
        native = {
            "resize_size": list(transforms.resize_size),
            "crop_size": list(transforms.crop_size),
            "interpolation": transforms.interpolation.value,
            "mean": list(transforms.mean), "std": list(transforms.std),
            "antialias": transforms.antialias,
        }
    return ModelSpec(
        architecture=architecture,
        weight_id=weight_id,
        native_preprocessing=native,
        encoder_channels=_ENCODER_CHANNELS[architecture],
        output_stride=32,
        gradcam_target_path="encoder." + _PARTIAL_PATHS[architecture][-1],
    )


class PretrainedClassifier(nn.Module):
    """A real spatial encoder, native pooling/normalization, and a fresh 200-logit head.

    Fine-tuning transitions update ``config`` to reflect the current policy. Frozen
    encoders retain autograd when inputs require gradients, so CAM callers may use
    ``images.requires_grad_(True)`` even in the fully frozen policy.
    """

    def __init__(self, config: ModelConfig, native: nn.Module, *, weights_loaded: bool) -> None:
        super().__init__()
        self.config = config
        self.spec = get_model_spec(config.architecture, config.weights)
        self.configured_weight_id = self.spec.weight_id
        self.weights_loaded = weights_loaded
        self.loaded_weight_id = self.spec.weight_id if weights_loaded else None
        if config.architecture == "convnextv2_base":
            if (not isinstance(native.head.pre_logits, nn.Identity)
                    or not isinstance(native.head.drop, nn.Dropout) or native.head.drop.p != 0):
                raise ValueError("Unsupported native ConvNeXtV2 head preprocessing")
            self.encoder = nn.Sequential(OrderedDict(
                stem=native.stem, stages=native.stages, norm_pre=native.norm_pre))
            self.pool = native.head.global_pool
            self.encoder_norm = native.head.norm
            channels = native.head.fc.in_features
        elif config.architecture == "convnext_tiny":
            self.pool = native.avgpool
            self.encoder = native.features
            self.encoder_norm = native.classifier[0]
            channels = native.classifier[2].in_features
        elif config.architecture == "efficientnet_b3":
            self.pool = native.avgpool
            self.encoder = native.features
            self.encoder_norm = nn.Identity()
            channels = native.classifier[1].in_features
        else:
            self.pool = native.avgpool
            self.encoder = nn.Sequential(OrderedDict(stem=native.stem, trunk_output=native.trunk_output))
            self.encoder_norm = nn.Identity()
            channels = native.fc.in_features
        if channels != self.spec.encoder_channels:
            raise ValueError("Encoder dimensions differ from the supported model specification")
        self.head = nn.Sequential(nn.Dropout(config.head_dropout), nn.Linear(channels, config.num_classes))
        self.set_fine_tuning(config.fine_tuning)

    def _partial_modules(self) -> tuple[nn.Module, ...]:
        return tuple(self.encoder.get_submodule(path) for path in _PARTIAL_PATHS[self.config.architecture])

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """The final spatial stage, resolved without duplicate module registration."""
        return self.get_submodule(self.spec.gradcam_target_path)

    def forward_features(self, images: Tensor) -> dict[str, Tensor]:
        """Return spatial NCHW features and pooled embeddings before head dropout."""
        spatial = self.encoder(images)
        pooled = self.encoder_norm(self.pool(spatial)).flatten(1)
        return {"spatial": spatial, "pooled": pooled}

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.forward_features(images)["pooled"])

    def set_fine_tuning(self, mode: str, *, freeze_bn_stats: bool | None = None) -> None:
        """Set frozen/partial/full masks and preserve frozen-stage evaluation state."""
        freeze_stats = self.config.freeze_bn_stats if freeze_bn_stats is None else freeze_bn_stats
        validate_fine_tuning(mode, freeze_stats)
        self.config = replace(self.config, fine_tuning=mode, freeze_bn_stats=freeze_stats)
        apply_parameter_policy(self.encoder, self.encoder_norm, self.head, self._partial_modules(), mode)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        apply_training_policy(
            self.encoder, self.encoder_norm, self._partial_modules(), self.config.fine_tuning,
            training=mode, freeze_bn_stats=self.config.freeze_bn_stats,
        )
        return self


def create_pretrained_model(
    config: ModelConfig, *, load_weights: bool = True, progress: bool = True,
) -> PretrainedClassifier:
    """Construct the selected CNN; propagate weight-loading failures unchanged.

    With ``load_weights=False``, construction is offline and no pretrained
    parameters are loaded. The timm dependency is imported only for ConvNeXtV2.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    if type(load_weights) is not bool or type(progress) is not bool:
        raise TypeError("load_weights and progress must be bools")
    if config.architecture == "convnextv2_base":
        import timm
        native = timm.create_model(V2BASE_WEIGHT_INFO["source_profile"], pretrained=False)
        if load_weights:
            native.load_state_dict(_load_pinned_state(V2BASE_WEIGHT_INFO, progress), strict=True)
        return PretrainedClassifier(config, native, weights_loaded=load_weights)
    is_fb = config.weights == FB22K_WEIGHT_ID
    native = getattr(models, config.architecture)(
        weights=_weight_enum(config.architecture) if load_weights and not is_fb else None,
        progress=progress,
    )
    if load_weights and is_fb:
        native.load_state_dict(convert_fb_convnext_tiny_state(_load_fb22k_state(progress)), strict=True)
    return PretrainedClassifier(config, native, weights_loaded=load_weights)


def convert_fb_convnext_tiny_state(raw: Mapping) -> OrderedDict:
    """Convert the pinned timm layout into Torchvision's identical Tiny architecture.

    All 182 FP32 tensors, including the discarded 1000-class head, are validated.
    Only layer-scale vectors change shape; every parameter value is preserved.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("checkpoint state must be a tensor mapping")
    layout = OrderedDict()

    def add(source, target, shape):
        layout[source] = (target, shape)

    add("stem.0.weight", "features.0.0.weight", (96, 3, 4, 4))
    add("stem.0.bias", "features.0.0.bias", (96,))
    for suffix in ("weight", "bias"):
        add("stem.1." + suffix, "features.0.1." + suffix, (96,))
    for stage, (channels, depth) in enumerate(zip((96, 192, 384, 768), (3, 3, 9, 3))):
        if stage:
            for suffix in ("weight", "bias"):
                add(f"stages.{stage}.downsample.0.{suffix}", f"features.{2 * stage}.0.{suffix}", (channels // 2,))
            add(f"stages.{stage}.downsample.1.weight", f"features.{2 * stage}.1.weight", (channels, channels // 2, 2, 2))
            add(f"stages.{stage}.downsample.1.bias", f"features.{2 * stage}.1.bias", (channels,))
        shapes = {
            "conv_dw.weight": (channels, 1, 7, 7), "conv_dw.bias": (channels,),
            "norm.weight": (channels,), "norm.bias": (channels,),
            "mlp.fc1.weight": (4 * channels, channels), "mlp.fc1.bias": (4 * channels,),
            "mlp.fc2.weight": (channels, 4 * channels), "mlp.fc2.bias": (channels,),
            "gamma": (channels,),
        }
        names = {"conv_dw": "block.0", "norm": "block.2", "mlp.fc1": "block.3", "mlp.fc2": "block.5"}
        for block in range(depth):
            for suffix, shape in shapes.items():
                if suffix == "gamma":
                    native_suffix = "layer_scale"
                else:
                    module, parameter = suffix.rsplit(".", 1)
                    native_suffix = names[module] + "." + parameter
                add(f"stages.{stage}.blocks.{block}.{suffix}", f"features.{2 * stage + 1}.{block}.{native_suffix}", shape)
    for suffix in ("weight", "bias"):
        add("head.norm." + suffix, "classifier.0." + suffix, (768,))
    add("head.fc.weight", "classifier.2.weight", (1000, 768))
    add("head.fc.bias", "classifier.2.bias", (1000,))
    if set(raw) != set(layout):
        raise ValueError(
            f"checkpoint state keys differ; missing={sorted(set(layout) - set(raw))!r}, "
            f"unexpected={sorted(set(raw) - set(layout), key=str)!r}"
        )
    converted = OrderedDict()
    for source, (target, shape) in layout.items():
        value = raw[source]
        if not isinstance(value, Tensor):
            raise TypeError(f"checkpoint value must be a tensor: {source}")
        if tuple(value.shape) != shape:
            raise ValueError(f"checkpoint tensor shape mismatch: {source}")
        if value.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor dtype must be FP32: {source}")
        if value.device.type != "cpu" or not torch.isfinite(value).all().item():
            raise ValueError(f"checkpoint tensor must contain finite CPU values: {source}")
        converted[target] = value.reshape(-1, 1, 1) if source.endswith(".gamma") else value
    return converted


def _load_fb22k_state(progress: bool):
    return _load_pinned_state(FB22K_WEIGHT_INFO, progress)


def load_safe_weight_state(path):
    """Read a CPU state mapping without permitting executable checkpoint objects."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_pinned_state(info, progress):
    path = Path(torch.hub.get_dir()) / "checkpoints" / info["filename"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            info["url"], str(path), hash_prefix=info["hash_prefix"], progress=progress,
        )
    if not path.is_file():
        raise ValueError(f"official checkpoint is not a file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != info["hash_prefix"]:
        raise ValueError(f"official checkpoint SHA-256 mismatch: {path}")
    return load_safe_weight_state(path)
