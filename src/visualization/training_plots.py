"""Training curves, confusion matrix visualization, and tuning comparison charts."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.training.trainer import EpochReport


def plot_training_curves(
    reports: Sequence[EpochReport],
    output_path: str | Path,
    *,
    title: str = "Training and Validation Metrics",
) -> None:
    """Plot Loss and Accuracy curves side-by-side across epochs."""
    epochs = [r.epoch for r in reports]
    train_loss = [r.train_loss for r in reports]
    val_loss = [r.val_loss for r in reports]
    train_acc = [r.train_accuracy * 100 for r in reports]
    val_acc = [r.val_accuracy * 100 for r in reports]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="royalblue", linewidth=2)
    ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="crimson", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.set_title("Loss vs. Epoch", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Accuracy curve
    ax2.plot(epochs, train_acc, "o-", label="Train Accuracy (%)", color="royalblue", linewidth=2)
    ax2.plot(epochs, val_acc, "s--", label="Val Accuracy (%)", color="forestgreen", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy vs. Epoch", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_summary(
    matrix: Tensor,
    output_path: str | Path,
    *,
    class_names: Sequence[str] | None = None,
    top_k: int = 10,
) -> None:
    """Plot full confusion matrix overview and top confused species pairs."""
    mat = matrix.detach().cpu().numpy()
    num_classes = mat.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})

    # 1. Macro 2D matrix heatmap
    im = ax1.imshow(mat, cmap="Blues", interpolation="nearest")
    ax1.set_title(f"Confusion Matrix ({num_classes}x{num_classes})", fontsize=12)
    ax1.set_xlabel("Predicted Class ID")
    ax1.set_ylabel("True Class ID")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 2. Extract top off-diagonal confusions
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, 0)
    confusions = []
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            count = int(off_diag[true_idx, pred_idx])
            if count > 0:
                true_label = class_names[true_idx] if class_names else f"Class {true_idx}"
                pred_label = class_names[pred_idx] if class_names else f"Class {pred_idx}"
                confusions.append((count, f"{true_label[:14]} → {pred_label[:14]}"))

    confusions.sort(key=lambda x: x[0], reverse=True)
    top_confusions = confusions[:top_k]

    if top_confusions:
        counts = [c[0] for c in reversed(top_confusions)]
        pairs = [c[1] for c in reversed(top_confusions)]
        y_pos = np.arange(len(pairs))
        ax2.barh(y_pos, counts, color="coral", edgecolor="black", alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(pairs, fontsize=9)
        ax2.set_xlabel("Misclassification Count")
        ax2.set_title(f"Top {len(top_confusions)} Most Confused Pairs", fontsize=12)
        ax2.grid(True, linestyle=":", alpha=0.6, axis="x")
    else:
        ax2.text(0.5, 0.5, "Zero misclassifications", ha="center", va="center")
        ax2.set_title("No Confused Pairs Found")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tuning_comparison(
    comparison_dict: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Plot bar chart comparing Grid Search vs Random Search."""
    methods = ["Grid Search", "Random Search"]
    grid_data = comparison_dict["grid_search"]
    rand_data = comparison_dict["random_search"]

    accuracies = [grid_data["best_val_accuracy"] * 100, rand_data["best_val_accuracy"] * 100]
    times = [grid_data["total_time_seconds"], rand_data["total_time_seconds"]]
    trials = [grid_data["num_trials"], rand_data["num_trials"]]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4))

    # Best Accuracy
    bars1 = ax1.bar(methods, accuracies, color=["royalblue", "mediumseagreen"], width=0.5)
    ax1.set_ylabel("Best Val Accuracy (%)")
    ax1.set_title("Best Accuracy Achieved")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom")

    # Total Time
    bars2 = ax2.bar(methods, times, color=["orange", "teal"], width=0.5)
    ax2.set_ylabel("Total Time (seconds)")
    ax2.set_title("Execution Time Taken")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.1f}s", ha="center", va="bottom")

    # Number of Trials
    bars3 = ax3.bar(methods, trials, color=["purple", "salmon"], width=0.5)
    ax3.set_ylabel("Trial Count")
    ax3.set_title("Number of Trials Explored")
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{int(yval)}", ha="center", va="bottom")

    plt.suptitle("Hyperparameter Tuning Method Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
