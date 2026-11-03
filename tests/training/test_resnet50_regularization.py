"""Regression checks for regularization without downloading data or weights."""

from copy import deepcopy
import json
from unittest.mock import MagicMock

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.resnet50_transforms import resnet50_transform
from src.data.transforms import classification_transform
from src.models.resnet50 import ResNet50Classifier
import src.training.resnet50_experiment as experiment
from src.training.resnet50_runner import load_config
from src.training.resnet50_training import EpochMetrics, train_one_epoch, validate_one_epoch


def test_regularized_validation_is_identical_and_training_is_seeded():
    image = Image.new("RGB", (281, 111), (120, 50, 20))
    expected = classification_transform(training=False)(image)
    actual = resnet50_transform(training=False, augmentation="mild")(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    transform = resnet50_transform(training=True, augmentation="mild")
    torch.manual_seed(42)
    first = transform(image)
    torch.manual_seed(42)
    second = transform(image)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32 and torch.isfinite(first).all()


def test_dropout_preserves_checkpoint_keys_and_is_disabled_for_evaluation():
    original = ResNet50Classifier(weights=None)
    regularized = ResNet50Classifier(weights=None, dropout=0.3)
    regularized.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.ones(1000)
    regularized.train()
    assert regularized.dropout.training
    assert (regularized.dropout(inputs) == 0).any()
    regularized.eval()
    torch.testing.assert_close(regularized.dropout(inputs), inputs, rtol=0, atol=0)


def test_label_smoothing_is_used_only_for_training():
    model = nn.Linear(2, 2)
    images = torch.eye(2)
    labels = torch.tensor([0, 1])
    loader = DataLoader([
        {"image": image, "label": label}
        for image, label in zip(images, labels)
    ], batch_size=2)
    with torch.no_grad():
        logits = model(images)
        smoothed = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1).item()
        plain = nn.functional.cross_entropy(logits, labels).item()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    training = train_one_epoch(model, loader, optimizer, torch.device("cpu"), label_smoothing=0.1)
    validation = validate_one_epoch(model, loader, torch.device("cpu"))
    assert abs(training.loss - smoothed) < 1e-6
    assert abs(validation.loss - plain) < 1e-6


class _SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer4 = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 200)
        self.stage = "head"

    def set_stage(self, stage):
        self.stage = stage
        self.encoder.requires_grad_(stage == "finetune")
        self.classifier.requires_grad_(True)
        self.zero_grad(set_to_none=True)


def test_early_stopping_preserves_last_and_best_checkpoints(tmp_path, monkeypatch):
    config = load_config("configs/resnet50_regularized.yaml")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "preparation.json").write_text(json.dumps({
        "manifest_sha256": "a" * 64,
        "class_names": [str(index) for index in range(200)],
        "source_revisions": {"train": "test-revision"},
    }), encoding="utf-8")
    config["data"]["prepared_dir"] = str(prepared)
    config["tracking"]["local_dir"] = str(tmp_path / "tracking")
    config["checkpoints"]["local_root"] = str(tmp_path / "models")
    config["training"]["head"]["epochs"] = 1
    config["training"]["finetune"]["early_stopping"] = {"patience": 2, "min_delta": 0.001}
    original = deepcopy(config)
    client = MagicMock()
    client.get_experiment_by_name.return_value.experiment_id = "test"
    client.create_run.return_value.info.run_id = "early-stop-check"
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda **kwargs: client)
    training = MagicMock(return_value=EpochMetrics(1.0, 0.5, 2))
    validation = MagicMock(side_effect=[
        EpochMetrics(loss, 0.5, 2) for loss in (1.5, 1.0, 0.9, 0.95, 0.96)
    ])
    monkeypatch.setattr(experiment, "train_one_epoch", training)
    monkeypatch.setattr(experiment, "validate_one_epoch", validation)
    loader = DataLoader([{"image": torch.zeros(2), "label": 0}] * 2, batch_size=2)
    run_dir = experiment.run_experiment(
        _SmallModel(), {"train": loader, "val": loader}, config, torch.device("cpu"),
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert summary["early_stopping"]["epoch"] == 4
    assert last["stage"] == "finetune" and last["epoch"] == 4
    assert best["stage"] == "head" and best["epoch"] == 1
    assert training.call_count == 5
    assert training.call_args.kwargs["label_smoothing"] == 0.1
    assert config == original
    client.set_terminated.assert_called_once_with("early-stop-check", status="FINISHED")
