"""Offline unit tests for training curves, confusion matrix, and tuning plots."""

from pathlib import Path
import tempfile

import torch

from src.training.trainer import EpochReport
from src.visualization.training_plots import (
    plot_confusion_summary,
    plot_training_curves,
    plot_tuning_comparison,
)


def test_plot_training_curves():
    reports = [
        EpochReport(epoch=1, train_loss=3.5, train_accuracy=0.1, val_loss=3.2, val_accuracy=0.15, learning_rate=1e-3),
        EpochReport(epoch=2, train_loss=2.8, train_accuracy=0.25, val_loss=2.5, val_accuracy=0.3, learning_rate=5e-4),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "curves.png"
        plot_training_curves(reports, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0


def test_plot_confusion_summary():
    # 5 classes with some diagonal and some confusion
    matrix = torch.tensor([
        [10, 2, 0, 0, 0],
        [1, 8, 1, 0, 0],
        [0, 0, 12, 0, 0],
        [0, 3, 0, 7, 0],
        [0, 0, 0, 0, 10],
    ])
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "confusion.png"
        plot_confusion_summary(matrix, out_png, class_names=[f"Bird_{i}" for i in range(5)])
        assert out_png.exists()
        assert out_png.stat().st_size > 0


def test_plot_tuning_comparison():
    comp = {
        "grid_search": {"best_val_accuracy": 0.42, "total_time_seconds": 120.0, "num_trials": 8},
        "random_search": {"best_val_accuracy": 0.45, "total_time_seconds": 65.0, "num_trials": 5},
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_png = Path(tmp_dir) / "tuning.png"
        plot_tuning_comparison(comp, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0
