"""Confusion-matrix classification metrics with explicit validation.

Both functions are pure and dependency-free beyond torch: callers own device
placement, batching, and any aggregation across batches. Inputs are validated
before any statistic is computed; invalid class ids or shapes raise instead of
producing silent counts. Classes with no predictions or no support contribute
0.0 to macro averages (``zero_division=0`` semantics).
"""

from torch import Tensor
import torch


def _validate_class_ids(values: Tensor, name: str, num_classes: int) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype == torch.bool or values.dtype.is_floating_point or values.dtype.is_complex:
        raise TypeError(f"{name} must have an integer dtype")
    if values.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    host = values.detach().to("cpu")
    lowest, highest = int(host.min()), int(host.max())
    if lowest < 0 or highest >= num_classes:
        raise ValueError(f"{name} contains ids outside [0, {num_classes})")
    return host


def confusion_matrix(predictions: Tensor, labels: Tensor, *, num_classes: int) -> Tensor:
    """Count predictions per class pair; rows are true classes, columns are predicted.

    Returns an int64 CPU tensor of shape ``(num_classes, num_classes)`` so
    callers may sum matrices across batches. Inputs are validated as equal-length,
    non-boolean, integer, one-dimensional, non-empty tensors whose ids lie in
    ``[0, num_classes)``.
    """
    if type(num_classes) is not int or isinstance(num_classes, bool) or num_classes < 1:
        raise ValueError("num_classes must be an integer >= 1")
    if not isinstance(predictions, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("predictions and labels must be torch.Tensors")
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have identical shapes")
    host_predictions = _validate_class_ids(predictions, "predictions", num_classes)
    host_labels = _validate_class_ids(labels, "labels", num_classes)
    pairs = host_labels * num_classes + host_predictions
    counts = torch.bincount(pairs, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).to(torch.int64)


def _validate_matrix(matrix: Tensor) -> Tensor:
    if not isinstance(matrix, Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square and two-dimensional")
    if matrix.dtype == torch.bool or matrix.dtype.is_floating_point or matrix.dtype.is_complex:
        raise TypeError("matrix must have an integer dtype")
    if matrix.shape[0] == 0:
        raise ValueError("matrix must not be empty")
    host = matrix.detach().to("cpu")
    if int(host.min()) < 0:
        raise ValueError("matrix counts must be non-negative")
    return host


def summarize_classification(matrix: Tensor) -> dict[str, float]:
    """Overall accuracy and macro precision/recall/F1 from one confusion matrix.

    The matrix follows :func:`confusion_matrix` orientation: rows are true
    classes, columns are predicted classes. A class with no predicted samples
    has precision 0.0 and a class with no true samples has recall 0.0; both
    enter the macro average with that value rather than being skipped.
    """
    host = _validate_matrix(matrix)
    num_classes = host.shape[0]
    total = int(host.sum())
    correct = int(host.diagonal().sum())
    true_positive = host.diagonal().to(torch.float64)
    false_positive = host.sum(dim=0).to(torch.float64) - true_positive
    false_negative = host.sum(dim=1).to(torch.float64) - true_positive
    precision = torch.where(
        true_positive + false_positive > 0,
        true_positive / (true_positive + false_positive).clamp(min=1),
        torch.zeros((), dtype=torch.float64),
    )
    recall = torch.where(
        true_positive + false_negative > 0,
        true_positive / (true_positive + false_negative).clamp(min=1),
        torch.zeros((), dtype=torch.float64),
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall).clamp(min=1e-12),
        torch.zeros((), dtype=torch.float64),
    )
    return {
        "accuracy": correct / total,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
