from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import torch
from torch import nn
import yaml
from src.resnet50.model import (
    INATURALIST_WEIGHTS, ResNet50Classifier, load_inaturalist_encoder,
)
import json
from PIL import Image
from src.resnet50 import model as prediction
from src.data.image_io import load_rgb
from src.resnet50.model import resnet50_transform
from copy import deepcopy
import numpy as np
from src.resnet50.model import RandomFitAndPad, resnet50_transform
from src.data.transforms import classification_transform
from src.resnet50.train import load_config


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
    import src.resnet50.train as runner

    config = runner.load_config("configs/resnet50/regularized.yaml")
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
    from src.resnet50.train import load_config

    config = load_config("configs/resnet50/regularized.yaml")
    config["model"].update({
        "weights": INATURALIST_WEIGHTS,
        "pretrained_checkpoint": "pretrained.pth.tar", "pretrained_sha256": "a" * 64,
    })
    config["model"][field] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_config(path)


def test_runner_defaults_to_selected_inaturalist_workflow(monkeypatch):
    import sys
    from pathlib import Path
    import src.resnet50.train as runner

    received = []

    def capture_config(path):
        received.append(path)
        raise ValueError("Stop before loading data or weights.")

    monkeypatch.setattr(sys, "argv", ["resnet50_runner", "--check-batch"])
    monkeypatch.setattr(runner, "load_config", capture_config)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert received == [Path("configs/resnet50/inaturalist.yaml")]


class TinyClassifier(nn.Module):
    """Small substitute; the actual selected ResNet is checked separately."""

    def __init__(self, num_classes, *, weights, dropout, finetune_scope):
        super().__init__()
        assert weights is None  # Prediction must not download pretrained weights.
        self.classifier = nn.Linear(3, num_classes)

    def forward(self, images):
        return self.classifier(images.mean(dim=(2, 3)))


@pytest.fixture
def saved_model(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction, "ResNet50Classifier", TinyClassifier)
    model = TinyClassifier(3, weights=None, dropout=0, finetune_scope="layer4")
    with torch.no_grad():
        model.classifier.weight.copy_(torch.eye(3))
        model.classifier.bias.copy_(torch.tensor([0.1, 0.3, -0.2]))
    config = {
        "model": {"num_classes": 3, "dropout": 0, "finetune_scope": "layer4"},
        "data": {"image_size": 32, "geometry": "fit", "normalization": "imagenet"},
        "data_provenance": {"class_names": ["Zebra Finch", "Blue Jay", "Robin"]},
    }
    payload = {"num_classes": 3, "model_state_dict": model.state_dict(), "config": config}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    image = tmp_path / "bird.png"
    with Image.new("RGBA", (19, 7), (40, 80, 120, 128)) as source:
        source.save(image)
    return checkpoint, image, payload, model


def test_roundtrip_preserves_scores_preprocessing_and_saved_label_order(saved_model):
    checkpoint, image, payload, original = saved_model
    before = checkpoint.read_bytes()
    with load_rgb({"path": str(image)}) as rgb:
        tensor = resnet50_transform(training=False, image_size=32, geometry="fit")(rgb)
    with torch.inference_mode():
        expected = original(tensor.unsqueeze(0)).softmax(1)[0]
    first = prediction.predict_image(image, checkpoint, top_k=3)
    assert first == prediction.predict_image(image, checkpoint, top_k=3)
    assert first["preprocessing"] == payload["config"]["data"]
    assert [row["confidence"] for row in first["predictions"]] == sorted(expected.tolist(), reverse=True)
    for row in first["predictions"]:
        assert row["class_name"] == payload["config"]["data_provenance"]["class_names"][row["class_id"]]
        assert row["confidence"] == pytest.approx(expected[row["class_id"]].item())
    restored, _ = prediction.load_classifier(checkpoint)
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("problem", ["labels", "normalization", "geometry", "state"])
def test_invalid_checkpoint_cannot_silently_change_predictions(saved_model, problem):
    checkpoint, _, payload, _ = saved_model
    if problem == "labels":
        payload["config"]["data_provenance"]["class_names"] = ["same"] * 3
    elif problem == "normalization":
        payload["config"]["data"]["normalization"] = "unknown"
    elif problem == "geometry":
        del payload["config"]["data"]["geometry"]
    else:
        del payload["model_state_dict"]["classifier.bias"]
    torch.save(payload, checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        prediction.load_classifier(checkpoint)


def test_nonfinite_scores_are_rejected(saved_model):
    checkpoint, image, payload, _ = saved_model
    payload["model_state_dict"]["classifier.bias"][0] = float("nan")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="nonfinite"):
        prediction.predict_image(image, checkpoint, top_k=3)


def test_cli_writes_json_and_does_not_overwrite_files(saved_model, tmp_path, capsys):
    checkpoint, image, _, _ = saved_model
    output = tmp_path / "prediction.json"
    args = ["--image", str(image), "--checkpoint", str(checkpoint), "--top-k", "3", "--output", str(output)]
    assert prediction.main(args) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text(encoding="utf-8"))
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        prediction.main(args)
    assert error.value.code == 2 and output.read_bytes() == before


def test_missing_image_and_invalid_top_k_fail_explicitly(saved_model):
    checkpoint, image, _, _ = saved_model
    with pytest.raises(FileNotFoundError):
        prediction.predict_image(image.with_name("missing.png"), checkpoint)
    for top_k in (0, True, 4):
        with pytest.raises(ValueError, match="top_k"):
            prediction.predict_image(image, checkpoint, top_k=top_k)


def test_cuda_request_does_not_silently_fall_back_to_cpu(saved_model, monkeypatch):
    checkpoint, _, _, _ = saved_model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        prediction.load_classifier(checkpoint, device="cuda")


@pytest.mark.parametrize("shape", [(121, 500), (500, 139), (500, 500)])
def test_random_fit_retains_image_edges_and_aspect_ratio(shape):
    image = Image.new("RGB", shape, (0, 0, 0))
    for corner, color in zip([(0, 0), (shape[0]-1, 0), (0, shape[1]-1), (shape[0]-1, shape[1]-1)],
                             [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        x, y = corner
        image.paste(color, (max(0, x-11), max(0, y-11), min(shape[0], x+12), min(shape[1], y+12)))
    torch.manual_seed(8)
    output = np.asarray(RandomFitAndPad(224)(image))
    ys, xs = np.where(np.any(output != 255, axis=2))
    width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    assert 190 <= max(width, height) <= 224
    assert abs(width / height - shape[0] / shape[1]) <= 2 / height
    red, green, blue = output[..., 0], output[..., 1], output[..., 2]
    assert ((red > 200) & (green < 50) & (blue < 50)).any()
    assert ((green > 200) & (red < 50) & (blue < 50)).any()
    assert ((blue > 200) & (red < 50) & (green < 50)).any()
    assert ((red > 200) & (green > 200) & (blue < 50)).any()
    assert output.shape == (224, 224, 3)


def test_training_replays_seed_and_varies_placement():
    image = Image.new("RGB", (300, 120), (50, 100, 150))
    transform = resnet50_transform(training=True, augmentation="spatial")
    torch.manual_seed(42)
    first = transform(image)
    torch.manual_seed(42)
    torch.testing.assert_close(first, transform(image), rtol=0, atol=0)
    torch.manual_seed(43)
    assert not torch.equal(first, transform(image))
    assert first.shape == (3, 224, 224) and torch.isfinite(first).all()


def test_validation_is_identical_and_trial_changes_only_augmentation():
    image = Image.new("RGB", (500, 139), (10, 20, 30))
    actual = resnet50_transform(training=False, augmentation="spatial")(image)
    torch.testing.assert_close(actual, classification_transform(training=False)(image), rtol=0, atol=0)
    baseline = load_config("configs/resnet50/regularized.yaml")
    trial = deepcopy(load_config("configs/resnet50/spatial.yaml"))
    trial["experiment"]["name"] = baseline["experiment"]["name"]
    trial["data"]["augmentation"] = baseline["data"]["augmentation"]
    assert trial == baseline
    with pytest.raises(ValueError, match="fit geometry"):
        resnet50_transform(training=True, geometry="crop", augmentation="spatial")


@pytest.mark.parametrize("scale", [0, -1, 1.1, True, float("nan")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="min_scale"):
        RandomFitAndPad(224, scale)
