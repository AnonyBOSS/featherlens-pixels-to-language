"""Visualization modules for EDA, training curves, and Grad-CAM."""

from .gradcam import (
    GradCAM,
    GradCAMResult,
    collect_gradcam_samples,
    denormalize_image,
    plot_gradcam_grid,
)

__all__ = [
    "GradCAM",
    "GradCAMResult",
    "collect_gradcam_samples",
    "denormalize_image",
    "plot_gradcam_grid",
]
