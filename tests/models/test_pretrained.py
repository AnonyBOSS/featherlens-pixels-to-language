"""Offline contracts for real Torchvision encoders; no weights are downloaded."""

from dataclasses import FrozenInstanceError, asdict
import importlib
import json

import pytest
import torch
from torch import nn
from torchvision import models


ARCHITECTURES = ["convnext_tiny", "efficientnet_b3", "regnet_y_3_2gf"]
EXPECTED = {
    "convnext_tiny": (768, "ConvNeXt_Tiny_Weights.IMAGENET1K_V1", 236, 224, "bilinear"),
    "efficientnet_b3": (1536, "EfficientNet_B3_Weights.IMAGENET1K_V1", 320, 300, "bicubic"),
    "regnet_y_3_2gf": (1512, "RegNet_Y_3_2GF_Weights.IMAGENET1K_V2", 232, 224, "bilinear"),
}


@pytest.fixture
def api():
    return importlib.import_module("src.models.pretrained")


@pytest.fixture(params=ARCHITECTURES)
def model(request, api):
    return api.create_pretrained_model(api.ModelConfig(request.param), load_weights=False)


def test_model_adapter_exists():
    assert importlib.util.find_spec("src.models.pretrained") is not None


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_validated_config_and_native_recipe_are_serializable(api, architecture):
    config = api.ModelConfig(architecture)
    spec = api.get_model_spec(architecture)
    channels, weight_id, resize, crop, interpolation = EXPECTED[architecture]
    assert config.weights == weight_id
    assert spec.weight_id == weight_id
    assert spec.encoder_channels == channels
    assert spec.output_stride == 32
    assert spec.native_preprocessing == {
        "resize_size": [resize], "crop_size": [crop], "interpolation": interpolation,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "antialias": True,
    }
    json.dumps(asdict(config))
    json.dumps(asdict(spec))
    with pytest.raises(FrozenInstanceError):
        config.num_classes = 10


@pytest.mark.parametrize("changes", [
    {"architecture": "resnet50"}, {"architecture": []}, {"weights": "DEFAULT"},
    {"weights": "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1"}, {"weights": 1},
    {"num_classes": 199}, {"num_classes": 200.0}, {"num_classes": True},
    {"fine_tuning": "last"}, {"fine_tuning": []}, {"head_dropout": -0.1},
    {"head_dropout": 1}, {"head_dropout": float("nan")}, {"head_dropout": True},
    {"head_dropout": "0.5"}, {"freeze_bn_stats": 1},
])
def test_invalid_config_fails_before_construction(api, changes):
    with pytest.raises((TypeError, ValueError)):
        api.ModelConfig(**({"architecture": "convnext_tiny"} | changes))


def test_features_and_logits_at_224_use_one_backbone_pass_and_separate_head(model):
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        features = model.forward_features(images)
        assert calls == [1]
        assert features["spatial"].shape == (2, EXPECTED[model.config.architecture][0], 7, 7)
        assert features["pooled"].shape == (2, EXPECTED[model.config.architecture][0])
        logits = model(images)
        assert calls == [1, 1]
        assert logits.shape == (2, 200)
        torch.testing.assert_close(logits, model.head(features["pooled"]))
    handle.remove()
    assert model.gradcam_target_layer is model.get_submodule(model.spec.gradcam_target_path)
    names = [name for name, _ in model.named_modules(remove_duplicate=False)]
    assert len(names) == len({id(module) for _, module in model.named_modules(remove_duplicate=False)})
    assert not model.weights_loaded
    assert model.loaded_weight_id is None
    assert model.configured_weight_id == model.spec.weight_id


def test_forward_features_does_not_invoke_head(model):
    def forbidden(*args):
        pytest.fail("Feature extraction must not invoke the classification head")
    handle = model.head.register_forward_hook(forbidden)
    with torch.no_grad():
        model.forward_features(torch.randn(1, 3, 64, 64))
    handle.remove()


def test_convnext_preserves_native_encoder_normalization(api):
    torch.manual_seed(42)
    native = models.convnext_tiny(weights=None).eval()
    torch.manual_seed(42)
    wrapped = api.create_pretrained_model(api.ModelConfig("convnext_tiny"), load_weights=False).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = native.classifier[1](native.classifier[0](native.avgpool(native.features(images))))
        torch.testing.assert_close(wrapped.forward_features(images)["pooled"], expected)
    assert list(wrapped.encoder_norm.parameters())


def test_head_dropout_is_configurable(api):
    model = api.create_pretrained_model(api.ModelConfig("efficientnet_b3", head_dropout=0.3), load_weights=False)
    assert isinstance(model.head[0], nn.Dropout)
    assert model.head[0].p == 0.3
    assert model.head[1].in_features == 1536
    assert model.head[1].out_features == 200


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("load_weights", [False, True])
def test_weight_loading_dispatches_exact_enum_without_download(api, architecture, load_weights, monkeypatch):
    weight_id = EXPECTED[architecture][1]
    enum_name, member = weight_id.split(".")
    weight = getattr(getattr(models, enum_name), member)
    calls = []
    # Only the external state-dict fetch is replaced; Torchvision builds the real model.
    def state_dict(*, progress, check_hash):
        calls.append((progress, check_hash))
        return getattr(models, architecture)(weights=None).state_dict()
    monkeypatch.setattr(weight, "get_state_dict", state_dict)
    model = api.create_pretrained_model(api.ModelConfig(architecture), load_weights=load_weights, progress=False)
    assert calls == ([(False, True)] if load_weights else [])
    assert model.weights_loaded is load_weights
    assert model.loaded_weight_id == (weight_id if load_weights else None)


def test_real_weight_loader_errors_propagate_without_random_fallback(api, monkeypatch):
    def unavailable(**kwargs):
        raise OSError("deliberately unavailable weights")
    monkeypatch.setattr(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "get_state_dict", unavailable)
    with pytest.raises(OSError, match="deliberately unavailable"):
        api.create_pretrained_model(api.ModelConfig("convnext_tiny"))


@pytest.mark.parametrize("argument", [{"load_weights": 0}, {"progress": 1}])
def test_factory_rejects_non_boolean_loader_flags(api, argument):
    with pytest.raises((TypeError, ValueError)):
        api.create_pretrained_model(api.ModelConfig("convnext_tiny"), **argument)
