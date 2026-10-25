"""Verify content-preserving geometry and unchanged evaluation preprocessing."""

from copy import deepcopy

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.resnet50_transforms import RandomFitAndPad, resnet50_transform
from src.data.transforms import classification_transform
from src.training.resnet50_runner import load_config


@pytest.mark.parametrize("shape", [(121, 500), (500, 139), (500, 500)])
def test_random_fit_retains_image_edges_and_aspect_ratio(shape):
    image = Image.new("RGB", shape, (0, 0, 0))
    for corner, color in zip([(0, 0), (shape[0]-1, 0), (0, shape[1]-1), (shape[0]-1, shape[1]-1)],
                             [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        x, y = corner
        image.paste(color, (max(0, x-11), max(0, y-11), min(shape[0], x+12), min(shape[1], y+12)))
    torch.manual_seed(8)
    output = np.asarray(RandomFitAndPad(224)(image))
    ys, xs = np.where(np.any(output != 255, axis=2))
    width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    assert 190 <= max(width, height) <= 224
    assert abs(width / height - shape[0] / shape[1]) <= 2 / height
    red, green, blue = output[..., 0], output[..., 1], output[..., 2]
    assert ((red > 200) & (green < 50) & (blue < 50)).any()
    assert ((green > 200) & (red < 50) & (blue < 50)).any()
    assert ((blue > 200) & (red < 50) & (green < 50)).any()
    assert ((red > 200) & (green > 200) & (blue < 50)).any()
    assert output.shape == (224, 224, 3)


def test_training_replays_seed_and_varies_placement():
    image = Image.new("RGB", (300, 120), (50, 100, 150))
    transform = resnet50_transform(training=True, augmentation="spatial")
    torch.manual_seed(42)
    first = transform(image)
    torch.manual_seed(42)
    torch.testing.assert_close(first, transform(image), rtol=0, atol=0)
    torch.manual_seed(43)
    assert not torch.equal(first, transform(image))
    assert first.shape == (3, 224, 224) and torch.isfinite(first).all()


def test_validation_is_identical_and_trial_changes_only_augmentation():
    image = Image.new("RGB", (500, 139), (10, 20, 30))
    actual = resnet50_transform(training=False, augmentation="spatial")(image)
    torch.testing.assert_close(actual, classification_transform(training=False)(image), rtol=0, atol=0)
    baseline = load_config("configs/resnet50_regularized.yaml")
    trial = deepcopy(load_config("configs/resnet50_spatial.yaml"))
    trial["experiment"]["name"] = baseline["experiment"]["name"]
    trial["data"]["augmentation"] = baseline["data"]["augmentation"]
    assert trial == baseline
    with pytest.raises(ValueError, match="fit geometry"):
        resnet50_transform(training=True, geometry="crop", augmentation="spatial")


@pytest.mark.parametrize("scale", [0, -1, 1.1, True, float("nan")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="min_scale"):
        RandomFitAndPad(224, scale)
