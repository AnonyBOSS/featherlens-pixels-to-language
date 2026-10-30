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
    {"activation": "gelu"}, {"activation": "sigmoid"}, {"activation": 123},
])
def test_invalid_config_fails_before_construction(api, changes):
    with pytest.raises((TypeError, ValueError)):
        api.ScratchConfig(**changes)


def test_leaky_relu_activation_config(api):
    config = api.ScratchConfig(activation="leaky_relu")
    model = api.create_scratch_model(config)
    assert isinstance(model.encoder.stem[2], torch.nn.LeakyReLU)
    assert model.encoder.stem[2].negative_slope == 0.1
    # Check forward pass
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 200)


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
    assert 2_000_000 < count < 4_000_000


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
    for name in ("head.1.weight", "encoder.stages.3.0.conv2.weight", "encoder.stem.0.weight"):
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


def test_simplified_6_block_architecture_structure(model):
    # Stem: single 3x3 conv (stride 2) + BN + ReLU + MaxPool2d (padding 1), width 32
    assert len(model.encoder.stem) == 4
    assert isinstance(model.encoder.stem[0], torch.nn.Conv2d)
    assert model.encoder.stem[0].out_channels == 32
    assert model.encoder.stem[0].stride == (2, 2)
    assert isinstance(model.encoder.stem[1], torch.nn.BatchNorm2d)
    assert isinstance(model.encoder.stem[2], torch.nn.ReLU)
    assert isinstance(model.encoder.stem[3], torch.nn.MaxPool2d)

    # 4 stages with blocks (2, 2, 1, 1) = 6 blocks total, widths (64, 128, 256, 256)
    assert len(model.encoder.stages) == 4
    blocks_per_stage = [len(stage) for stage in model.encoder.stages]
    assert blocks_per_stage == [2, 2, 1, 1]
    assert sum(blocks_per_stage) == 6
    stage_widths = [stage[0].conv1.out_channels for stage in model.encoder.stages]
    assert stage_widths == [64, 128, 256, 256]

    # SE blocks removed: no submodule has attribute 'se'
    for name, module in model.named_modules():
        assert not hasattr(module, "se"), f"Module {name} still has SE block"

    # Pooling: 3x3 MaxPool (stride 2) followed by AdaptiveAvgPool2d(1)
    assert isinstance(model.pool, torch.nn.Sequential)
    assert len(model.pool) == 2
    assert isinstance(model.pool[0], torch.nn.MaxPool2d)
    assert model.pool[0].kernel_size == 3
    assert model.pool[0].stride == 2
    assert model.pool[0].padding == 1
    assert isinstance(model.pool[1], torch.nn.AdaptiveAvgPool2d)

    # Exact parameter count: 2,875,368
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 2_875_368


def test_cls_yaml_matches_simplified_regularization_and_activation():
    from pathlib import Path
    import yaml
    from src.training.trainer import TrainingConfig

    config_path = Path("configs/cls.yaml")
    assert config_path.is_file()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["architecture"] == "scratch_resnet"
    assert cfg["model"]["activation"] == "relu"
    assert cfg["model"]["head_dropout"] == 0.4
    assert cfg["training"]["weight_decay"] == 0.0005

    # Verify configs can be parsed into validated dataclasses
    from src.models.scratch import ScratchConfig
    model_cfg = ScratchConfig(
        num_classes=cfg["model"]["num_classes"],
        head_dropout=cfg["model"]["head_dropout"],
        activation=cfg["model"]["activation"],
    )
    assert model_cfg.activation == "relu"
    assert model_cfg.head_dropout == 0.4

    train_cfg = TrainingConfig(
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        label_smoothing=cfg["training"]["label_smoothing"],
        seed=cfg["training"]["seed"],
        optimizer=cfg["training"]["optimizer"],
        amp=cfg["training"]["amp"],
    )
    assert train_cfg.weight_decay == 0.0005

