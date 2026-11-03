"""Numerical raw-score attribution and original-image overlay contracts."""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn


class TwoChannelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv2d(3, 2, 1, bias=False)
        self.head = nn.Linear(2, 200, bias=False)
        self.spec = SimpleNamespace(gradcam_target_path="spatial")
        with torch.no_grad():
            self.spatial.weight.zero_()
            self.spatial.weight[0, 0, 0, 0] = 1
            self.spatial.weight[1, 1, 0, 0] = 1
            self.head.weight.zero_()
            self.head.weight[0] = torch.tensor([1., 2.])
            self.head.weight[1] = torch.tensor([3., -1.])

    @property
    def gradcam_target_layer(self):
        return self.spatial

    def forward(self, images):
        return self.head(self.spatial(images).mean((2, 3)))


def images():
    return torch.tensor([[[[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]],
                          [[0., 0.], [0., 0.]]]])


def compute(*args, **kwargs):
    return importlib.import_module("src.evaluation.gradcam").compute_gradcam(*args, **kwargs)


def test_raw_requested_component_score_has_analytic_cam_and_preserves_state():
    model, sample = TwoChannelClassifier().train(), images()
    model.spatial.eval()  # Mixed train/eval flags must survive.
    states = [module.training for module in model.modules()]
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 7)
    before = [(p.detach().clone(), p.grad.clone()) for p in model.parameters()]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        result = compute(model, sample, target_class=1)
    # d(score_1)/d(activation) = [3/4, -1/4], then ReLU/min-max.
    torch.testing.assert_close(result["heatmap"], torch.tensor([[0., 3/11], [7/11, 1.]]))
    assert result["raw_cam_min"] == 0 and result["raw_cam_max"] == 2.75
    assert result["target_class"] == 1 and result["predicted_class"] == 0
    torch.testing.assert_close(result["logits"][:2], torch.tensor([7.5, 5.]))
    assert result["logits"].shape == (200,)
    assert result["heatmap"].dtype == result["logits"].dtype == torch.float32
    assert result["heatmap"].device.type == result["logits"].device.type == "cpu"
    assert result["finite"] and not result["degenerate"]
    assert result["layer"] == "spatial"
    assert [module.training for module in model.modules()] == states
    assert not model.spatial._forward_hooks
    assert not sample.requires_grad and sample.grad is None
    for parameter, (value, gradient) in zip(model.parameters(), before):
        torch.testing.assert_close(parameter, value)
        torch.testing.assert_close(parameter.grad, gradient)


def test_frozen_parameters_still_allow_input_autograd_and_default_target():
    model = TwoChannelClassifier().requires_grad_(False)
    result = compute(model, images())
    assert result["target_class"] == result["predicted_class"] == 0
    torch.testing.assert_close(result["heatmap"], torch.tensor([[1., 2/3], [1/3, 0.]]))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize("failure", ["forward", "logits", "gradient"])
def test_errors_restore_model_flags_and_remove_hook(failure):
    class Failing(TwoChannelClassifier):
        def forward(self, sample):
            activation = self.spatial(sample)
            if failure == "forward":
                raise RuntimeError("deliberate forward failure")
            if failure == "logits":
                return self.head(activation.mean((2, 3))) * float("nan")
            return self.head(activation.sqrt().mean((2, 3)))

    model = Failing().train()
    states = [module.training for module in model.modules()]
    with pytest.raises((ValueError, RuntimeError), match="failure|[Nn]onfinite"):
        compute(model, torch.zeros(1, 3, 2, 2))
    assert [module.training for module in model.modules()] == states
    assert not model.spatial._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("sample", [torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2)])
def test_degenerate_cam_is_explicit_zero_map(sample):
    result = compute(TwoChannelClassifier(), sample)
    assert result["degenerate"] and result["finite"]
    torch.testing.assert_close(result["heatmap"], torch.zeros(2, 2))


@pytest.mark.parametrize("target", [-1, 200, True, 1.5])
def test_invalid_target_fails_without_leaking_hooks(target):
    model = TwoChannelClassifier()
    with pytest.raises(ValueError, match="target_class"):
        compute(model, images(), target_class=target)
    assert not model.spatial._forward_hooks


@pytest.mark.parametrize("size,canvas_size,box", [
    ((5, 9), 9, (2, 0, 7, 9)),  # Portrait, with 2 pixels of side padding.
    ((9, 5), 9, (0, 2, 9, 7)),  # Landscape, with 2 pixels of top padding.
    ((3, 9), 8, (2, 0, 5, 8)),  # 2.667 rounds up; left/right padding is 2/3.
])
def test_overlay_removes_fit_padding_and_keeps_all_native_content(tmp_path, size, canvas_size, box):
    render = importlib.import_module("src.evaluation.gradcam").render_gradcam
    image = Image.new("RGB", size, (20, 40, 60))
    image.putpixel((0, 0), (255, 1, 2))
    image.putpixel((size[0]-1, size[1]-1), (3, 4, 255))
    heat = torch.zeros(canvas_size, canvas_size)
    left, top, right, bottom = box
    heat[top:bottom, left:right] = 1
    path = render(image, heat, canvas_size, output_path=tmp_path / "overlay.png", alpha=1)
    with Image.open(path) as output:
        assert output.size == size
        # Every native pixel receives the same red endpoint; padding was removed.
        assert np.unique(np.asarray(output).reshape(-1, 3), axis=0).shape[0] == 1
        red, green, blue = output.getpixel((0, 0))
        assert red > green and red > blue
    path = render(image, heat, canvas_size, output_path=tmp_path / "original.png", alpha=0)
    with Image.open(path) as output:
        assert np.array_equal(np.asarray(output), np.asarray(image))
