"""Visualization modules for EDA, training curves, confusion matrices, and Grad-CAM."""

from .gradcam import (
    GradCAM,
    GradCAMResult,
    collect_gradcam_samples,
    denormalize_image,
    plot_gradcam_grid,
)
from .training_plots import (
    plot_confusion_summary,
    plot_training_curves,
    plot_tuning_comparison,
)

__all__ = [
    "GradCAM",
    "GradCAMResult",
    "collect_gradcam_samples",
    "denormalize_image",
    "plot_gradcam_grid",
    "plot_confusion_summary",
    "plot_training_curves",
    "plot_tuning_comparison",
]
