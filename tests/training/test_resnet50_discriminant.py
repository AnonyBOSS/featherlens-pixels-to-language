"""Check the constrained classifier against independent distance predictions."""

import pytest
import torch
from torch.nn import functional as F

from src.training.resnet50_head_refit import fit_discriminant_head


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
