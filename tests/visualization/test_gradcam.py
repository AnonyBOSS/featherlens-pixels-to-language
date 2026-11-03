"""Offline unit tests for Grad-CAM explainability."""

from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.scratch import ScratchConfig, create_scratch_model
from src.visualization.gradcam import (
    GradCAM,
    collect_gradcam_samples,
    denormalize_image,
    plot_gradcam_grid,
)


def test_gradcam_generates_normalized_heatmap_on_scratch_resnet():
    torch.manual_seed(42)
    model = create_scratch_model(ScratchConfig())
    cam = GradCAM(model)

    image = torch.randn(1, 3, 224, 224)
    heatmap, pred_class, confidence = cam.generate_heatmap(image)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min()
    assert heatmap.max() <= 1.0
    assert 0 <= pred_class < 200
    assert 0.0 <= confidence <= 1.0
    cam.remove_hooks()


def test_denormalize_image_returns_valid_rgb():
    tensor = torch.zeros(3, 100, 100)
    rgb = denormalize_image(tensor)
    assert rgb.shape == (100, 100, 3)
    assert 0.0 <= rgb.min() <= rgb.max() <= 1.0


def test_collect_and_plot_gradcam_samples():
    torch.manual_seed(42)
    model = create_scratch_model(ScratchConfig())

    # Create dummy dataset where some predictions are correct and some wrong
    x = torch.randn(6, 3, 224, 224)
    # Predict first to craft correct and wrong labels
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    y = preds.clone()
    y[0] = (y[0] + 1) % 200  # Force index 0 to be wrong

    def collate(batch):
        return {
            "image": torch.stack([b[0] for b in batch]),
            "label": torch.tensor([b[1] for b in batch]),
        }

    loader = DataLoader(TensorDataset(x, y), batch_size=2, collate_fn=collate)

    samples = collect_gradcam_samples(
        model,
        loader,
        device="cpu",
        class_names=[f"Bird_{i}" for i in range(200)],
        max_correct=1,
        max_wrong=1,
    )
    assert len(samples) >= 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "gradcam_test.png"
        plot_gradcam_grid(samples, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0
