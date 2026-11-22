import pytest
import torch
from torch import nn
from src.resnet50.head_refit import (
    calibrate_logit_scale,
    classification_metrics,
    collect_features,
    fit_regularized_head,
    fold_feature_standardization,
)
from torch.nn import functional as F
from src.resnet50.head_refit import fit_discriminant_head, fit_flip_augmented_head


def test_folded_layer_matches_standardized_predictions():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(9, 4, generator=generator)
    mean = torch.tensor([3., -2., 0., 1.])
    std = torch.tensor([0.5, 2., 1., 4.])
    weight = torch.randn(3, 4, generator=generator)
    bias = torch.randn(3, generator=generator)
    raw_weight, raw_bias = fold_feature_standardization(weight, bias, mean, std, 2.3)
    expected = 2.3 * nn.functional.linear((x - mean) / std, weight, bias)
    torch.testing.assert_close(nn.functional.linear(x, raw_weight, raw_bias), expected)


def test_validation_cannot_change_fitted_weights_without_calibration():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        x = torch.tensor([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
        y = torch.tensor([0, 0, 1, 1])
        kwargs = dict(num_classes=2, l2_strength=0.1, device=torch.device("cpu"), calibrate=False)
        first = fit_regularized_head(x, y, x, y, **kwargs)
        second = fit_regularized_head(x, y, x + 1000, 1 - y, **kwargs)
        torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
        torch.testing.assert_close(first.bias, second.bias, rtol=0, atol=0)
        assert first.logit_scale == 1.0
        assert first.train_metrics["accuracy"] == 1.0
        assert torch.isfinite(first.weight).all()  # Constant feature column is supported.
    finally:
        torch.set_num_threads(previous_threads)


def test_calibration_preserves_predictions_and_reduces_validation_nll():
    scores = torch.tensor([[10., 0.], [10., 0.], [0., 10.], [0., 10.]])
    labels = torch.tensor([0, 0, 1, 0])
    scale = calibrate_logit_scale(scores, labels)
    assert 0 < scale < 1
    assert torch.equal(scores.argmax(1), (scores * scale).argmax(1))
    assert nn.functional.cross_entropy(scores * scale, labels) < nn.functional.cross_entropy(scores, labels)


def test_metrics_include_all_classes_and_unsmoothed_loss():
    logits = torch.tensor([[3., 0., 0.], [0., 3., 0.], [0., 3., 0.]])
    labels = torch.tensor([0, 1, 2])
    values = classification_metrics(logits, labels)
    assert values["samples"] == 3
    assert values["accuracy"] == pytest.approx(2 / 3)
    assert values["macro_f1"] == pytest.approx((1 + 2 / 3 + 0) / 3)
    assert values["loss"] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())


@pytest.mark.parametrize("strength", [True, 0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_regularization_rejected(strength):
    with pytest.raises(ValueError, match="l2_strength"):
        fit_regularized_head(
            torch.eye(2), torch.arange(2), torch.eye(2), torch.arange(2),
            num_classes=2, l2_strength=strength, device=torch.device("cpu"),
        )


def test_feature_collection_flips_only_when_explicitly_requested():
    from torch.utils.data import DataLoader

    class Encoder(nn.Module):
        def encode(self, images):
            return images.flatten(1)

    model = Encoder()
    images = torch.arange(3 * 3 * 2 * 4, dtype=torch.float32).reshape(3, 3, 2, 4)
    loader = DataLoader([{"image": image, "label": index} for index, image in enumerate(images)], batch_size=2)
    original, labels = collect_features(model, loader, torch.device("cpu"))
    flipped, flip_labels = collect_features(model, loader, torch.device("cpu"), horizontal_flip=True)
    torch.testing.assert_close(original, images.flatten(1), rtol=0, atol=0)
    torch.testing.assert_close(flipped, images.flip(-1).flatten(1), rtol=0, atol=0)
    assert labels.tolist() == flip_labels.tolist() == [0, 1, 2]
    assert not model.training and not flipped.requires_grad


def test_two_view_refit_exports_reloadable_checkpoint_without_test_access(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from PIL import Image
    import mlflow.tracking
    import src.data.classification as classification
    import src.data.preparation as preparation
    import src.data.source as source
    import src.resnet50.model as resnet
    from src.resnet50.head_refit import refit_checkpoint

    class TinyModel(nn.Module):
        def __init__(self, num_classes, **kwargs):
            super().__init__()
            self.encoder = nn.Conv2d(3, 2, 1, bias=False)
            self.classifier = nn.Linear(2, num_classes)

        def set_stage(self, stage):
            self.encoder.requires_grad_(False)

        def encode(self, images):
            return self.encoder(images).mean((2, 3))

    class Dataset:
        def __init__(self, _source, _rows, *, partition, transform):
            assert partition in ("train", "val")
            self.transform = transform
            self.labels = [0, 0, 1, 1] if partition == "train" else [0, 1]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            label = self.labels[index]
            with Image.new("RGB", (9, 5), (40 + label * 100 + index * 5, 70 + index * 10, 90)) as image:
                return {"image": self.transform(image), "label": label}

    class Client:
        def __init__(self, **kwargs): pass
        def get_experiment_by_name(self, name): return SimpleNamespace(experiment_id="experiment")
        def create_run(self, *args, **kwargs): return SimpleNamespace(info=SimpleNamespace(run_id="unit-refit"))
        def log_param(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def log_artifact(self, *args, **kwargs): pass
        def set_terminated(self, *args, **kwargs): pass

    provenance = {"manifest_sha256": "fixture", "class_names": ["Bird A", "Bird B"], "source_revisions": {"train": "pinned"}}
    prepared = tmp_path / "preparation"
    prepared.mkdir()
    (prepared / "preparation.json").write_text(json.dumps(provenance), encoding="utf-8")
    config = {
        "model": {"num_classes": 2},
        "data": {"prepared_dir": str(prepared), "image_size": 8, "geometry": "fit", "normalization": "imagenet", "batch_size": 2},
        "data_provenance": provenance,
        "tracking": {"local_dir": str(tmp_path / "tracking")},
        "checkpoints": {"local_root": str(tmp_path / "runs")},
    }
    original = TinyModel(2)
    path = tmp_path / "source.pt"
    torch.save({"num_classes": 2, "config": config, "model_state_dict": original.state_dict()}, path)
    before = path.read_bytes()
    calls = []

    def load_split(partition):
        calls.append(partition)
        assert partition == "train"  # Canonical validation is also from the official train source.
        return SimpleNamespace(class_names=provenance["class_names"])

    monkeypatch.setattr(source, "load_cub_split", load_split)
    monkeypatch.setattr(preparation, "load_classification_manifest", lambda _: [])
    monkeypatch.setattr(classification, "CubClassificationDataset", Dataset)
    monkeypatch.setattr(resnet, "ResNet50Classifier", TinyModel)
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Client)
    destination = refit_checkpoint(path, None, torch.device("cpu"), method="discriminant", shrinkage=0.05,
                                   image_size=16, batch_size=2, train_horizontal_flip=True)
    payload = torch.load(destination, weights_only=True)
    restored = TinyModel(2)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    assert payload["config"]["data"]["image_size"] == 16
    assert payload["head_refit"]["encoder_training_image_size"] == 8
    assert payload["head_refit"]["fit_feature_vectors"] == 8
    assert payload["metrics"]["train"]["samples"] == 4
    assert payload["metrics"]["val"]["samples"] == 2
    assert payload["head_refit"]["train_views"] == "original_plus_horizontal_flip"
    assert calls == ["train"] and path.read_bytes() == before
    for key, value in original.encoder.state_dict().items():
        torch.testing.assert_close(restored.encoder.state_dict()[key], value, rtol=0, atol=0)


def example():
    x = torch.tensor([[-3., -1.], [-2., 1.], [1., 0.], [3., 2.], [-1., 4.], [1., 5.]])
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    xv = torch.tensor([[-2.5, 0.], [2., 1.], [0., 4.], [0., 1.]])
    yv = torch.tensor([0, 1, 2, 1])
    return x, y, xv, yv


def test_full_shrinkage_matches_euclidean_nearest_class_mean():
    x, y, xv, yv = example()
    fit = fit_discriminant_head(x, y, xv, yv, num_classes=3,
                               shrinkage=1.0, device=torch.device("cpu"), calibrate=False)
    means = torch.stack([x[y == index].mean(0) for index in range(3)])
    expected = torch.cdist(xv, means).argmin(1)
    actual = F.linear(xv, fit.weight, fit.bias).argmax(1)
    assert torch.equal(actual, expected)
    assert fit.logit_scale == 1.0


def test_validation_labels_and_statistics_cannot_fit_discriminant_weights():
    x, y, xv, yv = example()
    kwargs = dict(num_classes=3, shrinkage=0.5, device=torch.device("cpu"), calibrate=False)
    first = fit_discriminant_head(x, y, xv, yv, **kwargs)
    second = fit_discriminant_head(x, y, xv + 100, (yv + 1) % 3, **kwargs)
    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    torch.testing.assert_close(first.bias, second.bias, rtol=0, atol=0)


def test_rank_deficient_features_supported_by_positive_shrinkage():
    x, y, xv, yv = example()
    x = torch.cat([x, x, torch.ones(6, 1)], dim=1)
    xv = torch.cat([xv, xv, torch.ones(4, 1)], dim=1)
    fit = fit_discriminant_head(x, y, xv, yv, num_classes=3,
                               shrinkage=0.1, device=torch.device("cpu"))
    assert torch.isfinite(fit.weight).all() and torch.isfinite(fit.bias).all()
    assert fit.train_metrics["accuracy"] == 1.0


@pytest.mark.parametrize("strength", [0, -1, 1.1, True, float("nan"), None])
def test_invalid_shrinkage_rejected(strength):
    with pytest.raises(ValueError, match="shrinkage"):
        fit_discriminant_head(*example(), num_classes=3,
                              shrinkage=strength, device=torch.device("cpu"))


def test_zero_within_class_variation_is_rejected():
    x = torch.tensor([[0., 0.], [0., 0.], [1., 1.], [1., 1.]])
    y = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="within-class variation"):
        fit_discriminant_head(x, y, x, y, num_classes=2,
                              shrinkage=0.5, device=torch.device("cpu"))


def test_two_view_fit_matches_combined_training_statistics_and_clean_metrics():
    x, y, xv, yv = example()
    flipped = x + torch.tensor([[0.2, -0.4], [-0.1, 0.3], [0.4, 0.2], [0.1, -0.3], [-0.5, 0.1], [0.3, 0.2]])
    fit = fit_flip_augmented_head(x, y, xv, yv, flipped, num_classes=3, shrinkage=0.05, calibrate=False)
    reference = fit_discriminant_head(torch.cat((x, flipped)), torch.cat((y, y)), xv, yv,
                                     num_classes=3, shrinkage=0.05, device=torch.device("cpu"), calibrate=False)
    torch.testing.assert_close(fit.weight, reference.weight, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fit.bias, reference.bias, rtol=1e-5, atol=1e-5)
    assert fit.train_metrics["samples"] == len(y)  # Never report the two views as independent images.
    assert fit.val_metrics["samples"] == len(yv)
    assert fit.train_metrics["accuracy"] == pytest.approx((F.linear(x, fit.weight, fit.bias).argmax(1) == y).double().mean())


def test_augmented_statistics_cannot_use_validation_labels_or_features():
    x, y, xv, yv = example()
    kwargs = dict(num_classes=3, shrinkage=0.05, calibrate=False)
    first = fit_flip_augmented_head(x, y, xv, yv, x + 0.1, **kwargs)
    second = fit_flip_augmented_head(x, y, xv + 100, (yv + 1) % 3, x + 0.1, **kwargs)
    torch.testing.assert_close(first.weight, second.weight, rtol=0, atol=0)
    torch.testing.assert_close(first.bias, second.bias, rtol=0, atol=0)


def test_augmented_feature_rows_must_match_original_training_shape():
    x, y, xv, yv = example()
    with pytest.raises(ValueError, match="flipped_features"):
        fit_flip_augmented_head(x, y, xv, yv, x[:-1], num_classes=3, shrinkage=0.05)
