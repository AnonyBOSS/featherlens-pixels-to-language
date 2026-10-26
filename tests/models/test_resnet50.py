from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
import yaml

from src.models.resnet50 import (
    INATURALIST_WEIGHTS, ResNet50Classifier, load_inaturalist_encoder,
)


def test_model_output():
    model = ResNet50Classifier(weights=None)
    model.eval()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        predictions = model(images)

    assert predictions.shape == (2, 200)


def test_training():
    model = ResNet50Classifier(weights=None)
    model.set_stage("head")
    model.train()

    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 199])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad()
    predictions = model(images)
    loss = loss_fn(predictions, labels)

    assert torch.isfinite(loss).item()

    loss.backward()
    optimizer.step()

    after = model.classifier.weight

    assert not torch.equal(before, after)


def test_save_load(tmp_path):
    model = ResNet50Classifier(weights=None)
    model.eval()

    images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        original_predictions = model(images)

    checkpoint_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint_path)

    new_model = ResNet50Classifier(weights=None)
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    new_model.load_state_dict(state_dict)
    new_model.eval()

    with torch.no_grad():
        new_predictions = new_model(images)

    torch.testing.assert_close(original_predictions, new_predictions)


@pytest.fixture(scope="module")
def inaturalist_checkpoint(tmp_path_factory):
    """Local tensors only: these checks never download external weights."""
    model = ResNet50Classifier(weights=None)
    state = dict(model.encoder.state_dict())
    # Expanded views preserve the official 10,000-class shapes without an
    # unnecessary 82 MB allocation for a classifier that must be discarded.
    state["fc.weight"] = torch.zeros(1).expand(10000, 2048)
    state["fc.bias"] = torch.zeros(1).expand(10000)
    path = tmp_path_factory.mktemp("inat") / "pretrained.pth.tar"
    torch.save({"state_dict": state, "epoch": 100}, path)
    return path, sha256(path.read_bytes()).hexdigest(), state


def test_inaturalist_loads_encoder_and_preserves_cub_head(inaturalist_checkpoint, tmp_path):
    path, digest, state = inaturalist_checkpoint
    model = ResNet50Classifier(weights=None)
    head = {name: tensor.clone() for name, tensor in model.classifier.state_dict().items()}
    load_inaturalist_encoder(model, path, digest)
    for name, tensor in model.encoder.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    for name, tensor in model.classifier.state_dict().items():
        torch.testing.assert_close(tensor, head[name], rtol=0, atol=0)
    assert model.classifier.out_features == 200
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())

    # A trained-model export is self-contained: it needs neither the original
    # external checkpoint nor a model download to restore predictions.
    saved = tmp_path / "cub.pt"
    torch.save(model.state_dict(), saved)
    restored = ResNet50Classifier(weights=None)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        images = torch.randn(2, 3, 64, 64)
        torch.testing.assert_close(model(images), restored(images), rtol=0, atol=0)


def test_inaturalist_rejects_wrong_hash_before_reading_checkpoint(tmp_path):
    path = tmp_path / "not_a_checkpoint"
    path.write_bytes(b"invalid torch checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        load_inaturalist_encoder(ResNet50Classifier(weights=None), path, "0" * 64)


@pytest.mark.parametrize("problem", ["missing", "shape", "extra", "duplicate"])
def test_inaturalist_rejects_incompatible_state(inaturalist_checkpoint, tmp_path, problem):
    _, _, original = inaturalist_checkpoint
    state = dict(original)
    if problem == "missing":
        del state["layer4.2.conv3.weight"]
    elif problem == "shape":
        state["fc.weight"] = torch.zeros(1, 2048)
    elif problem == "extra":
        state["unexpected"] = torch.zeros(1)
    else:
        state["module.conv1.weight"] = state["conv1.weight"]
    path = tmp_path / "bad.pth.tar"
    torch.save({"state_dict": state}, path)
    digest = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_inaturalist_encoder(ResNet50Classifier(weights=None), path, digest)


def test_inaturalist_accepts_distributed_prefix(inaturalist_checkpoint, tmp_path):
    _, _, state = inaturalist_checkpoint
    path = tmp_path / "distributed.pth.tar"
    torch.save({"state_dict": {"module." + name: tensor for name, tensor in state.items()}}, path)
    model = ResNet50Classifier(weights=None)
    load_inaturalist_encoder(model, path, sha256(path.read_bytes()).hexdigest())
    torch.testing.assert_close(model.encoder.conv1.weight, state["conv1.weight"], rtol=0, atol=0)


def test_inaturalist_runner_uses_no_imagenet_weights(tmp_path, monkeypatch):
    import src.training.resnet50_runner as runner

    config = runner.load_config("configs/resnet50_regularized.yaml")
    config["model"].update({
        "weights": INATURALIST_WEIGHTS,
        "pretrained_checkpoint": str(tmp_path / "pretrained.pth.tar"),
        "pretrained_sha256": "a" * 64,
    })
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert runner.load_config(config_path) == config
    monkeypatch.setattr(runner, "load_classification_manifest", lambda path: [])
    monkeypatch.setattr(runner, "load_cub_split", lambda split: SimpleNamespace(class_names=range(200)))
    monkeypatch.setattr(runner, "CubClassificationDataset", MagicMock())
    monkeypatch.setattr(runner, "make_classification_loaders", lambda *args, **kwargs: {})
    constructor = MagicMock()
    loader = MagicMock()
    monkeypatch.setattr(runner, "ResNet50Classifier", constructor)
    monkeypatch.setattr(runner, "load_inaturalist_encoder", loader)
    runner.prepare_training(config, torch.device("cpu"))
    assert constructor.call_args.kwargs["weights"] is None
    loader.assert_called_once_with(
        constructor.return_value, tmp_path / "pretrained.pth.tar", "a" * 64,
    )


@pytest.mark.parametrize("field,value", [
    ("pretrained_checkpoint", ""), ("pretrained_sha256", ""),
    ("pretrained_sha256", "not-a-hash"),
])
def test_inaturalist_configuration_requires_path_and_hash(tmp_path, field, value):
    from src.training.resnet50_runner import load_config

    config = load_config("configs/resnet50_regularized.yaml")
    config["model"].update({
        "weights": INATURALIST_WEIGHTS,
        "pretrained_checkpoint": "pretrained.pth.tar", "pretrained_sha256": "a" * 64,
    })
    config["model"][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_config(path)
