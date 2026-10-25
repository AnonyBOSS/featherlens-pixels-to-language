"""Offline contracts for confusion-matrix metrics; pure tensors, no model."""

import importlib

import pytest
import torch


@pytest.fixture
def api():
    return importlib.import_module("src.evaluation.metrics")


def test_confusion_matrix_counts_follow_row_true_column_predicted_orientation(api):
    predictions = torch.tensor([0, 1, 1, 2])
    labels = torch.tensor([0, 1, 2, 2])
    matrix = api.confusion_matrix(predictions, labels, num_classes=3)
    assert matrix.dtype == torch.int64
    assert matrix.device.type == "cpu"
    assert torch.equal(matrix, torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]))
    assert int(matrix.sum()) == 4


def test_matrices_sum_across_batches_like_one_full_pass(api):
    first_predictions = torch.tensor([0, 1])
    second_predictions = torch.tensor([1, 2])
    labels = torch.tensor([0, 1, 2, 2])
    combined = api.confusion_matrix(torch.cat([first_predictions, second_predictions]), labels, num_classes=3)
    split = (api.confusion_matrix(first_predictions, labels[:2], num_classes=3)
             + api.confusion_matrix(second_predictions, labels[2:], num_classes=3))
    assert torch.equal(combined, split)


def test_summary_matches_hand_computed_macro_scores(api):
    predictions = torch.tensor([0, 1, 1, 2])
    labels = torch.tensor([0, 1, 2, 2])
    summary = api.summarize_classification(api.confusion_matrix(predictions, labels, num_classes=3))
    assert summary["accuracy"] == pytest.approx(0.75)
    assert summary["macro_precision"] == pytest.approx(5 / 6)
    assert summary["macro_recall"] == pytest.approx(5 / 6)
    assert summary["macro_f1"] == pytest.approx(7 / 9)


def test_classes_without_predictions_or_support_score_zero_not_skipped(api):
    predictions = torch.tensor([0, 0])
    labels = torch.tensor([0, 1])
    summary = api.summarize_classification(api.confusion_matrix(predictions, labels, num_classes=2))
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["macro_precision"] == pytest.approx(0.25)
    assert summary["macro_recall"] == pytest.approx(0.5)
    assert summary["macro_f1"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("changes", [
    {"predictions": [0, 1]}, {"labels": "x"},
    {"predictions": torch.tensor([[0], [1]])},
    {"predictions": torch.tensor([0, 1.0])},
    {"predictions": torch.tensor([True, False])},
    {"predictions": torch.tensor([], dtype=torch.int64)},
    {"labels": torch.tensor([0, 3])},
    {"labels": torch.tensor([0, -1])},
    {"num_classes": 0}, {"num_classes": True}, {"num_classes": 1.0}, {"num_classes": "3"},
])
def test_invalid_confusion_matrix_inputs_are_rejected(api, changes):
    values = {
        "predictions": torch.tensor([0, 1]),
        "labels": torch.tensor([0, 1]),
        "num_classes": 2,
    } | changes
    with pytest.raises((TypeError, ValueError)):
        api.confusion_matrix(**values)


@pytest.mark.parametrize("changes", [
    {"matrix": torch.tensor([1, 1])},
    {"matrix": torch.ones(2, 3, dtype=torch.int64)},
    {"matrix": torch.ones(2, 2)},
    {"matrix": torch.ones(0, 0, dtype=torch.int64)},
    {"matrix": torch.tensor([[1, -1], [0, 2]])},
    {"matrix": "cm"},
])
def test_invalid_summary_inputs_are_rejected(api, changes):
    with pytest.raises((TypeError, ValueError)):
        api.summarize_classification(**changes)
