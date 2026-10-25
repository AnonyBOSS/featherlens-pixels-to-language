"""Verify frozen-feature fitting and inference normalization equivalence."""

import pytest
import torch
from torch import nn

from src.training.resnet50_head_refit import (
    calibrate_logit_scale,
    classification_metrics,
    fit_regularized_head,
    fold_feature_standardization,
)


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
