"""Offline contracts for the from-scratch baseline; no downloads and no pretrained state."""

from dataclasses import FrozenInstanceError, asdict
import importlib
import json

import pytest
import torch


@pytest.fixture
def api():
    return importlib.import_module("src.models.scratch")


@pytest.fixture
def model(api):
    torch.manual_seed(42)
    return api.create_scratch_model(api.ScratchConfig())


def test_scratch_module_exists():
    assert importlib.util.find_spec("src.models.scratch") is not None


def test_spec_and_config_are_serializable_and_frozen(api):
    spec = api.get_scratch_spec()
    assert spec.architecture == "scratch_resnet"
    assert spec.encoder_channels == 256
    assert spec.output_stride == 32
    assert spec.gradcam_target_path == "encoder.stages.3"
    json.dumps(asdict(spec))
    config = api.ScratchConfig()
    json.dumps(asdict(config))
    with pytest.raises(FrozenInstanceError):
        config.num_classes = 10


@pytest.mark.parametrize("changes", [
    {"num_classes": 199}, {"num_classes": 200.0}, {"num_classes": True},
    {"head_dropout": -0.1}, {"head_dropout": 1}, {"head_dropout": float("nan")},
    {"head_dropout": True}, {"head_dropout": "0.5"},
])
def test_invalid_config_fails_before_construction(api, changes):
    with pytest.raises((TypeError, ValueError)):
        api.ScratchConfig(**changes)


def test_features_and_logits_at_224_use_one_encoder_pass_and_separate_head(model):
    model.eval()
    images = torch.randn(2, 3, 224, 224)
    calls = []
    handle = model.encoder.register_forward_hook(lambda *args: calls.append(1))
    with torch.no_grad():
        features = model.forward_features(images)
        assert calls == [1]
        assert features["spatial"].shape == (2, 256, 7, 7)
        assert features["pooled"].shape == (2, 256)
        logits = model(images)
        assert calls == [1, 1]
        assert logits.shape == (2, 200)
        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits, model.head(features["pooled"]))
    handle.remove()
    assert model.gradcam_target_layer is model.get_submodule(model.spec.gradcam_target_path)
    assert model.gradcam_target_layer is model.encoder.stages[3]
    names = [name for name, _ in model.named_modules(remove_duplicate=False)]
    assert len(names) == len({id(module) for _, module in model.named_modules(remove_duplicate=False)})


def test_forward_features_does_not_invoke_head(model):
    def forbidden(*args):
        pytest.fail("Feature extraction must not invoke the classification head")
    handle = model.head.register_forward_hook(forbidden)
    with torch.no_grad():
        model.forward_features(torch.randn(1, 3, 64, 64))
    handle.remove()


def test_parameter_count_is_in_the_planned_scratch_range(model):
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 3_000_000 < count < 8_000_000


def test_seeded_construction_is_reproducible(api):
    torch.manual_seed(42)
    first = api.create_scratch_model(api.ScratchConfig())
    torch.manual_seed(42)
    second = api.create_scratch_model(api.ScratchConfig())
    for (name_one, tensor_one), (name_two, tensor_two) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        assert name_one == name_two
        torch.testing.assert_close(tensor_one, tensor_two)


def test_backward_produces_finite_gradients_from_head_to_first_conv(model):
    loss = model(torch.randn(2, 3, 64, 64)).pow(2).sum()
    loss.backward()
    for name in ("head.1.weight", "encoder.stages.3.1.conv2.weight", "encoder.stem.0.weight"):
        gradient = model.get_parameter(name).grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_batchnorm_updates_running_stats_only_in_train_mode(model):
    model.eval()
    fixed = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        reference = model(fixed)
        torch.testing.assert_close(model(fixed), reference)
    model.train()
    before = model.encoder.stem[1].running_mean.clone()
    with torch.no_grad():
        model(fixed)
    assert not torch.equal(model.encoder.stem[1].running_mean, before)


def test_head_dropout_is_configurable(api):
    model = api.create_scratch_model(api.ScratchConfig(head_dropout=0.3))
    assert isinstance(model.head[0], torch.nn.Dropout)
    assert model.head[0].p == 0.3
    assert model.head[1].in_features == 256
    assert model.head[1].out_features == 200


def test_factory_rejects_non_config(api):
    with pytest.raises(TypeError):
        api.create_scratch_model("scratch")
