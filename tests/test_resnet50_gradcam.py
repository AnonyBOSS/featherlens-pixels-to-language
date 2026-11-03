"""Verify spatial class attribution, frozen-model safety and Grad-CAM output."""

from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn

from src.visualization import resnet50_gradcam as explanation


class SpatialClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, 1, bias=False)
        self.head = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.features.weight.zero_()
            self.features.weight[0, 0, 0, 0] = 1
            self.features.weight[1, 1, 0, 0] = 1
            self.head.weight.copy_(torch.eye(2))
            self.head.bias.zero_()
        self.encoder = SimpleNamespace(layer4=[self.features])
        self.eval().requires_grad_(False)

    def forward(self, images):
        return self.head(self.features(images).mean(dim=(2, 3)))


def spatial_input():
    images = torch.zeros(1, 3, 4, 4)
    images[0, 0, :2, :2] = 2
    images[0, 1, 2:, 2:] = 4
    return images


def test_each_class_highlights_its_own_known_region_with_frozen_parameters():
    model, images = SpatialClassifier(), spatial_input()
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for class_id, region in [(0, (slice(0, 2), slice(0, 2))), (1, (slice(2, 4), slice(2, 4)))]:
        heatmap, logits, selected = explanation.gradcam(model, images, model.features, class_id=class_id)
        expected = torch.zeros(4, 4)
        expected[region] = 1
        torch.testing.assert_close(heatmap, expected, rtol=0, atol=0)
        assert selected == class_id and logits.shape == (1, 2)
        assert not heatmap.requires_grad and not logits.requires_grad
    _, _, selected = explanation.gradcam(model, images, model.features)
    assert selected == 1
    assert not images.requires_grad and images.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_uses_logit_gradient_even_when_softmax_is_saturated():
    model, images = SpatialClassifier(), spatial_input()
    before, _, _ = explanation.gradcam(model, images, model.features, class_id=0)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([1000., -1000.]))
    with torch.no_grad():
        after, _, _ = explanation.gradcam(model, images, model.features, class_id=0)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_no_positive_support_returns_finite_zero_map():
    model = SpatialClassifier()
    with torch.no_grad():
        model.head.weight.neg_()
    heatmap, _, _ = explanation.gradcam(model, spatial_input(), model.features, class_id=0)
    assert torch.isfinite(heatmap).all() and torch.count_nonzero(heatmap) == 0


@pytest.mark.parametrize("class_id", [-1, True, 2])
def test_invalid_class_and_failure_do_not_leave_hooks(class_id):
    model = SpatialClassifier()
    before = len(model.features._forward_hooks)
    with pytest.raises(ValueError, match="class_id"):
        explanation.gradcam(model, spatial_input(), model.features, class_id=class_id)
    assert len(model.features._forward_hooks) == before


def test_unused_target_layer_is_rejected_and_hook_removed():
    model, unused = SpatialClassifier(), nn.Identity()
    with pytest.raises(ValueError, match="exactly once"):
        explanation.gradcam(model, spatial_input(), unused)
    assert len(unused._forward_hooks) == 0


def test_training_and_inference_modes_cannot_silently_disable_gradcam():
    model = SpatialClassifier()
    with torch.inference_mode(), pytest.raises(ValueError, match="autograd"):
        explanation.gradcam(model, spatial_input(), model.features)
    model.train()
    with pytest.raises(ValueError, match="eval"):
        explanation.gradcam(model, spatial_input(), model.features)


def test_cli_renders_metadata_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    model = SpatialClassifier()
    config = {
        "data": {"image_size": 8, "geometry": "fit", "normalization": "imagenet"},
        "data_provenance": {"class_names": ["Red bird", "Green bird"]},
    }
    monkeypatch.setattr(explanation, "load_classifier", lambda *args, **kwargs: (model, config))
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"stand-in for loader")
    image = tmp_path / "bird.png"
    with Image.new("RGB", (9, 5), (200, 80, 20)) as rgb:
        rgb.save(image)
    output = tmp_path / "maps" / "example.png"
    args = ["--image", str(image), "--checkpoint", str(checkpoint), "--output", str(output),
            "--class-id", "1", "--true-class-id", "1"]
    assert explanation.main(args) == 0
    import json
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["explained_class_name"] == "Green bird"
    assert report["correct"] == (report["predicted_class_id"] == 1)
    assert report["weights_updated"] is False
    with Image.open(output) as figure:
        assert figure.format == "PNG" and figure.width > figure.height
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        explanation.main(args)
    assert error.value.code == 1 and output.read_bytes() == before
    assert len(model.features._forward_hooks) == 0
