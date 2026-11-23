import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.resnet50.model import ResNet50Classifier
import src.resnet50.training as training
from copy import deepcopy
import json
from unittest.mock import MagicMock
from PIL import Image
from src.resnet50.model import resnet50_transform
from src.data.transforms import classification_transform
import src.resnet50.training as experiment
from src.resnet50.train import load_config
from src.resnet50.training import EpochMetrics, train_one_epoch, validate_one_epoch


def test_last_block_scope_updates_only_selected_encoder_parameters():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ResNet50Classifier(weights=None, finetune_scope="last_block")
        model.set_stage("finetune")
        model.train()
        assert not model.encoder.layer4[0].training
        assert model.encoder.layer4[-1].training
        assert all(not layer.training for layer in model.encoder.modules() if isinstance(layer, nn.BatchNorm2d))
        assert sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) == 4462592
        optimizer = training.build_optimizer(
            model, classifier_learning_rate=0.0001,
            encoder_learning_rate=0.00001, weight_decay=0.01,
        )
        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert grouped == {id(p) for p in model.parameters() if p.requires_grad}
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        last_weight = model.encoder.layer4[-1].conv3.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 199]))
        loss.backward()
        optimizer.step()
        assert not torch.equal(last_weight, model.encoder.layer4[-1].conv3.weight)
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert parameter.grad is None
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        model.set_stage("head")
        assert not any(p.requires_grad for p in model.encoder.parameters())
    finally:
        torch.set_num_threads(previous_threads)


def test_mixup_uses_both_targets_and_weighted_accuracy(monkeypatch):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    images = 2 * torch.eye(2)
    labels = torch.tensor([0, 1])
    loader = DataLoader([
        {"image": image, "label": label}
        for image, label in zip(images, labels)
    ], batch_size=2)
    monkeypatch.setattr(training.np.random, "beta", lambda *args: 0.25)
    monkeypatch.setattr(training.torch, "randperm", lambda count, **kwargs: torch.tensor([1, 0], **kwargs))
    mixed = 0.25 * images + 0.75 * images.flip(0)
    with torch.no_grad():
        logits = model(mixed)
        expected = (
            0.25 * nn.functional.cross_entropy(logits, labels)
            + 0.75 * nn.functional.cross_entropy(logits, labels.flip(0))
        ).item()
    metrics = training.train_one_epoch(
        model, loader, torch.optim.SGD(model.parameters(), lr=0.0),
        torch.device("cpu"), mixup_alpha=0.2,
    )
    assert abs(metrics.loss - expected) < 1e-6
    assert metrics.accuracy == 0.75
    assert metrics.samples == 2


@pytest.mark.parametrize("alpha", [-0.1, float("nan"), float("inf"), True])
def test_invalid_mixup_fails_before_training(alpha):
    model = nn.Linear(2, 2)
    model.eval()
    with pytest.raises(ValueError, match="mixup_alpha"):
        training.train_one_epoch(
            model, [], torch.optim.SGD(model.parameters(), lr=0.01),
            torch.device("cpu"), mixup_alpha=alpha,
        )
    assert not model.training


def test_scope_preserves_state_dictionary_compatibility():
    original = ResNet50Classifier(weights=None)
    selective = ResNet50Classifier(weights=None, finetune_scope="last_block")
    selective.load_state_dict(original.state_dict(), strict=True)
    assert original.state_dict().keys() == selective.state_dict().keys()


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
    config = load_config("configs/resnet50/regularized.yaml")
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
