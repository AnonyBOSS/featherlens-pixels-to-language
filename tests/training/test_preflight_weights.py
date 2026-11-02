"""Offline weight contracts; synthetic checkpoints never constitute official evidence."""

from collections import OrderedDict
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path

import pytest
import torch
from torchvision import models

from src.models.pretrained import ModelConfig, PretrainedClassifier

ARCHITECTURES = ("convnext_tiny", "efficientnet_b3", "regnet_y_3_2gf")
EXPECTED = {
    "convnext_tiny": ("ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "convnext_tiny-983f1562.pth", "983f1562"),
    "efficientnet_b3": ("EfficientNet_B3_Weights.IMAGENET1K_V1", "efficientnet_b3_rwightman-b3899882.pth", "b3899882"),
    "regnet_y_3_2gf": ("RegNet_Y_3_2GF_Weights.IMAGENET1K_V2", "regnet_y_3_2gf-9180c971.pth", "9180c971"),
}


@pytest.fixture
def api():
    return importlib.import_module("src.training.weights")


@pytest.fixture(params=ARCHITECTURES, scope="module")
def synthetic(request):
    """Real adapter and independent random native state; no pretrained claim."""
    architecture = request.param
    native = getattr(models, architecture)(weights=None)
    state = OrderedDict((key, value.detach().clone()) for key, value in native.state_dict().items())
    adapter = PretrainedClassifier(ModelConfig(architecture), native, weights_loaded=False)
    return adapter, state


def test_weight_verification_module_exists():
    assert importlib.util.find_spec("src.training.weights") is not None


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_resolve_exact_pinned_official_identity_without_io(api, architecture, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("metadata must not access cache"))
    info = api.resolve_weight_info(ModelConfig(architecture))
    weight_id, filename, prefix = EXPECTED[architecture]
    assert info["weight_id"] == weight_id
    assert info["url"] == "https://download.pytorch.org/models/" + filename
    assert info["filename"] == filename
    assert info["hash_prefix"] == prefix
    assert info["architecture"] == architecture
    assert info["torch_version"] == torch.__version__
    import torchvision
    assert info["torchvision_version"] == torchvision.__version__
    json.dumps(info)


def test_invalid_configuration_and_switches_fail_before_io(api, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("invalid arguments accessed cache"))
    with pytest.raises(TypeError, match="ModelConfig"):
        api.resolve_weight_info({"architecture": "convnext_tiny"})
    for kwargs in ({"offline": 1}, {"progress": 1}):
        with pytest.raises(TypeError, match="bool"):
            api.prepare_official_weight(ModelConfig("convnext_tiny"), **kwargs)


def test_missing_offline_checkpoint_does_not_download_or_make_cache(api, tmp_path, monkeypatch):
    cache = tmp_path / "not-created"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache))
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("offline network access"))
    with pytest.raises(FileNotFoundError, match="offline"):
        api.prepare_official_weight(ModelConfig("convnext_tiny"), offline=True)
    assert not cache.exists()


def _synthetic_cache(monkeypatch, tmp_path, payload):
    """Replace only the external URL/cache seams with a hash-identifiable fixture."""
    digest = hashlib.sha256(payload).hexdigest()
    filename = f"synthetic-{digest[:8]}.pth"
    weight = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    monkeypatch.setattr(weight.value, "url", "https://download.pytorch.org/models/" + filename)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    return tmp_path / "checkpoints" / filename, digest


def test_existing_valid_cache_records_full_digest_without_download(api, tmp_path, monkeypatch):
    payload = b"synthetic cache bytes, not pretrained weights"
    path, digest = _synthetic_cache(monkeypatch, tmp_path, payload)
    path.parent.mkdir()
    path.write_bytes(payload)
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("duplicate download"))
    info = api.prepare_official_weight(ModelConfig("efficientnet_b3"), offline=True)
    assert info["cache_path"] == str(path)
    assert info["sha256"] == digest
    assert info["size_bytes"] == len(payload)
    assert info["hash_verified"] is True
    json.dumps(info)


def test_supported_download_uses_hash_then_reuses_one_artifact(api, tmp_path, monkeypatch):
    payload = b"synthetic downloaded bytes, not official evidence"
    path, digest = _synthetic_cache(monkeypatch, tmp_path, payload)
    calls = []
    def download(url, destination, *, hash_prefix, progress):
        assert url == "https://download.pytorch.org/models/" + path.name
        assert hash_prefix == digest[:8]
        assert progress is False
        calls.append(destination)
        Path(destination).write_bytes(payload)
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    first = api.prepare_official_weight(ModelConfig("efficientnet_b3"), progress=False)
    second = api.prepare_official_weight(ModelConfig("efficientnet_b3"), offline=True)
    assert first == second
    assert len(calls) == 1
    assert first["sha256"] == digest


@pytest.mark.parametrize("cached", [True, False])
def test_corrupt_bytes_fail_hash_verification_without_fallback(api, tmp_path, monkeypatch, cached):
    path, _ = _synthetic_cache(monkeypatch, tmp_path, b"expected synthetic bytes")
    if cached:
        path.parent.mkdir()
        path.write_bytes(b"corrupt")
    def download(url, destination, **kwargs):
        if cached:
            pytest.fail("corrupt cache must not silently redownload")
        Path(destination).write_bytes(b"corrupt")
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    with pytest.raises(ValueError, match="SHA-256"):
        api.prepare_official_weight(ModelConfig("efficientnet_b3"))


def test_download_failure_propagates(api, tmp_path, monkeypatch):
    _synthetic_cache(monkeypatch, tmp_path, b"synthetic")
    def unavailable(*args, **kwargs):
        raise OSError("deliberate unavailable download")
    monkeypatch.setattr(torch.hub, "download_url_to_file", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        api.prepare_official_weight(ModelConfig("efficientnet_b3"))


def test_all_native_retained_tensors_match_exactly(api, synthetic):
    model, state = synthetic
    proof = api.verify_backbone_state(model, state)
    retained = {key: value for key, value in model.state_dict().items() if not key.startswith("head.")}
    assert proof["status"] == "verified"
    assert proof["retained_tensor_count"] == len(retained)
    assert proof["parameter_tensor_count"] == sum(not key.startswith("head.") for key, _ in model.named_parameters())
    assert proof["buffer_tensor_count"] == len(list(model.named_buffers()))
    assert len(proof["excluded_native_head_keys"]) == 2
    json.dumps(proof)


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "value", "type"])
def test_incompatible_retained_state_is_rejected(api, synthetic, corruption):
    model, native_state = synthetic
    state = OrderedDict(native_state)
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    elif corruption == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = state[key].reshape(-1)[:1]
    elif corruption == "dtype":
        state[key] = state[key].double()
    elif corruption == "value":
        state[key] = state[key].clone()
        state[key].view(-1)[0] += 1
    else:
        state[key] = [1]
    with pytest.raises((TypeError, ValueError), match="missing|unexpected|shape|dtype|value|tensor"):
        api.verify_backbone_state(model, state)


def test_buffers_and_convnext_normalization_are_verified(api, synthetic):
    model, native_state = synthetic
    target = "classifier.0.weight" if model.config.architecture == "convnext_tiny" else next(
        key for key in native_state if key.endswith("running_mean")
    )
    state = OrderedDict(native_state)
    state[target] = state[target].clone()
    state[target].view(-1)[0] += 1
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, state)


def test_only_the_expected_native_1000_class_head_is_excluded(api, synthetic):
    model, native_state = synthetic
    head_weight = {"convnext_tiny": "classifier.2.weight", "efficientnet_b3": "classifier.1.weight", "regnet_y_3_2gf": "fc.weight"}[model.config.architecture]
    state = OrderedDict(native_state)
    state[head_weight] = torch.zeros(200, native_state[head_weight].shape[1])
    with pytest.raises(ValueError, match="1000|shape"):
        api.verify_backbone_state(model, state)
    state = OrderedDict(native_state)
    state.pop(head_weight)
    with pytest.raises(ValueError, match="missing"):
        api.verify_backbone_state(model, state)


def test_new_head_must_have_200_trainable_outputs(api, synthetic):
    model, state = synthetic
    model.head[1].weight.requires_grad_(False)
    try:
        with pytest.raises(ValueError, match="trainable"):
            api.verify_backbone_state(model, state)
    finally:
        model.head[1].weight.requires_grad_(True)
    original = model.head[1]
    model.head[1] = torch.nn.Linear(original.in_features, 201)
    try:
        with pytest.raises(ValueError, match="200"):
            api.verify_backbone_state(model, state)
    finally:
        model.head[1] = original


def test_verified_builder_uses_real_factory_and_safe_cpu_checkpoint(api, tmp_path, monkeypatch):
    native = models.efficientnet_b3(weights=None)
    state = native.state_dict()
    stream = io.BytesIO()
    torch.save(state, stream)
    path, digest = _synthetic_cache(monkeypatch, tmp_path, stream.getvalue())
    path.parent.mkdir()
    path.write_bytes(stream.getvalue())
    del native, stream
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("unexpected network"))
    real_load = torch.load
    loads = []
    def safe_load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", safe_load)
    model, proof = api.build_verified_pretrained_model(ModelConfig("efficientnet_b3"), offline=True, progress=False)
    assert model.weights_loaded is True
    assert model.loaded_weight_id == "EfficientNet_B3_Weights.IMAGENET1K_V1"
    assert proof["sha256"] == digest
    assert proof["backbone"]["status"] == "verified"
    assert proof["head"] == {"out_features": 200, "trainable": True}
    assert any(item.get("weights_only") is True and item.get("map_location") == "cpu" for item in loads)
    json.dumps(proof)


@pytest.mark.parametrize("payload_kind", ["not_mapping", "unsafe_pickle", "wrong_state"])
def test_builder_rejects_invalid_checkpoint_without_fallback(api, tmp_path, monkeypatch, payload_kind):
    stream = io.BytesIO()
    payload = [torch.zeros(1)] if payload_kind == "not_mapping" else (
        Path("unsafe-global") if payload_kind == "unsafe_pickle" else {"bad": torch.zeros(1)}
    )
    torch.save(payload, stream)
    path, _ = _synthetic_cache(monkeypatch, tmp_path, stream.getvalue())
    path.parent.mkdir()
    path.write_bytes(stream.getvalue())
    if payload_kind != "wrong_state":
        monkeypatch.setattr(api, "create_pretrained_model", lambda *a, **k: pytest.fail("invalid checkpoint reached factory"))
    with pytest.raises((TypeError, ValueError, RuntimeError), match="checkpoint|state|missing|load|tensor"):
        api.build_verified_pretrained_model(ModelConfig("efficientnet_b3"), offline=True)


def test_even_one_ulp_retained_value_difference_is_rejected(api, synthetic):
    model, native_state = synthetic
    state = OrderedDict(native_state)
    key = next(iter(state))
    state[key] = state[key].clone()
    scalar = state[key].view(-1)
    scalar[0] = torch.nextafter(scalar[0], torch.full_like(scalar[0], float("inf")))
    assert torch.allclose(state[key], native_state[key])
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, state)


@pytest.mark.parametrize("synthetic", ["efficientnet_b3", "regnet_y_3_2gf"], indirect=True)
def test_integer_batch_counter_is_verified(api, synthetic):
    model, native_state = synthetic
    key = next(key for key in native_state if key.endswith("num_batches_tracked"))
    state = OrderedDict(native_state)
    state[key] = state[key] + 1
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, state)


def test_builder_rejects_factory_state_disagreement(api, tmp_path, monkeypatch):
    native = models.efficientnet_b3(weights=None)
    state = native.state_dict()
    stream = io.BytesIO()
    torch.save(state, stream)
    path, _ = _synthetic_cache(monkeypatch, tmp_path, stream.getvalue())
    path.parent.mkdir()
    path.write_bytes(stream.getvalue())
    key = next(key for key in state if key.endswith("running_mean"))
    state[key] = state[key] + 1
    def mismatched_download(*, progress, check_hash):
        return state
    monkeypatch.setattr(models.EfficientNet_B3_Weights.IMAGENET1K_V1, "get_state_dict", mismatched_download)
    with pytest.raises(ValueError, match="value"):
        api.build_verified_pretrained_model(ModelConfig("efficientnet_b3"), offline=True)


def test_unsupported_filename_hash_fails_before_cache_io(api, monkeypatch):
    monkeypatch.setattr(models.EfficientNet_B3_Weights.IMAGENET1K_V1.value, "url", "https://download.pytorch.org/models/invalid.pth")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("unsupported artifact accessed cache"))
    with pytest.raises(ValueError, match="hash prefix"):
        api.prepare_official_weight(ModelConfig("efficientnet_b3"), offline=True)


def test_fb_profile_resolves_full_hash_and_immutable_revision_without_io(api, monkeypatch):
    from src.models.pretrained import FB22K_WEIGHT_ID
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("metadata accessed cache"))
    info = api.resolve_weight_info(ModelConfig("convnext_tiny", weights=FB22K_WEIGHT_ID))
    assert info["weight_id"] == FB22K_WEIGHT_ID
    assert info["revision"] == "32c741ba106f9e062174ef829354a16719c3dcac"
    assert info["hash_prefix"] == "c1253ad2d4a751cb0b1fb92a850f99e8f426a701d6e7600b03a490557d77bdcd"
    assert "/resolve/" + info["revision"] + "/" in info["url"]
    assert info["hash_prefix"] in info["filename"]


def test_v2base_pinned_metadata_and_safe_cpu_state_reader(api, tmp_path, monkeypatch):
    from src.models.pretrained import V2BASE_WEIGHT_ID, load_safe_weight_state
    save_file = pytest.importorskip("safetensors.torch").save_file
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("metadata accessed cache"))
    info = api.resolve_weight_info(ModelConfig("convnextv2_base"))
    assert info["weight_id"] == V2BASE_WEIGHT_ID
    assert info["revision"] == "bea9dc139d5981aff6ce4d6d9690086e45ad8b56"
    assert info["hash_prefix"] == "972bcbb8d9965eca7a2f72d9592f54dfc890f9e6aef920cea4269963f46f2f42"
    assert info["source_profile"] == "convnextv2_base.fcmae_ft_in22k_in1k_384"
    path = tmp_path / "small.safetensors"
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    save_file(state, str(path))
    loaded = load_safe_weight_state(path)
    assert loaded["weight"].device.type == "cpu"
    assert torch.equal(loaded["weight"], state["weight"])


def test_v2base_proof_checks_all_retained_tensors_and_native_head(api):
    timm = pytest.importorskip("timm")
    native = timm.create_model("convnextv2_base", pretrained=False)
    state = native.state_dict()
    model = PretrainedClassifier(ModelConfig("convnextv2_base"), native, weights_loaded=False)
    proof = api.verify_backbone_state(model, state)
    assert proof["retained_tensor_count"] == 378
    assert proof["excluded_native_head_keys"] == ["head.fc.bias", "head.fc.weight"]
    bad = dict(state)
    bad["head.norm.bias"] = bad["head.norm.bias"] + 1
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, bad)
    bad = dict(state)
    bad["head.fc.weight"] = bad["head.fc.weight"][:200]
    with pytest.raises(ValueError, match="1000|shape"):
        api.verify_backbone_state(model, bad)
