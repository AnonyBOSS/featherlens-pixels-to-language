"""Check evaluation metrics, exact coverage and absence of weight updates."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.resnet50_evaluation import evaluate_loader, metrics_from_confusion


def test_metrics_treat_rows_as_truth_and_include_missing_predictions():
    matrix = torch.tensor([[2, 0, 0], [1, 1, 0], [0, 2, 0]])
    metrics = metrics_from_confusion(matrix)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_precision"] == pytest.approx((2/3 + 1/3 + 0)/3)
    assert metrics["macro_recall"] == pytest.approx((1 + 0.5 + 0)/3)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 0.4 + 0)/3)
    assert metrics["per_class"][2]["support"] == 2


def samples():
    return [
        {"image": torch.tensor([3., 0.]), "label": 0, "image_id": "a"},
        {"image": torch.tensor([0., 3.]), "label": 1, "image_id": "b"},
        {"image": torch.tensor([0., 2.]), "label": 0, "image_id": "c"},
    ]


def test_evaluation_counts_final_partial_batch_without_updates():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    original = model.weight.detach().clone()
    metrics, matrix, predictions = evaluate_loader(
        model, DataLoader(samples(), batch_size=2), device=torch.device("cpu"),
        num_classes=2, expected_samples=3,
    )
    assert not model.training and model.weight.grad is None
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    assert metrics["samples"] == 3 and metrics["correct"] == 2
    assert metrics["accuracy"] == pytest.approx(2/3)
    assert matrix.tolist() == [[1, 1], [0, 1]]
    assert [row["image_id"] for row in predictions] == ["a", "b", "c"]


def test_evaluation_rejects_missing_or_repeated_images():
    kwargs = dict(device=torch.device("cpu"), num_classes=2, expected_samples=4)
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_loader(nn.Identity(), DataLoader(samples(), batch_size=2), **kwargs)
    repeated = samples() + [samples()[0]]
    with pytest.raises(ValueError, match="Repeated"):
        evaluate_loader(nn.Identity(), DataLoader(repeated, batch_size=2), **kwargs)


def test_nonfinite_predictions_are_rejected():
    class InvalidModel(nn.Module):
        def forward(self, x):
            return x * float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        evaluate_loader(InvalidModel(), DataLoader(samples(), batch_size=3),
                        device=torch.device("cpu"), num_classes=2, expected_samples=3)
