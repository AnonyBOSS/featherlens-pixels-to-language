"""Grad-CAM visualization for convolutional neural network explainability."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class GradCAMResult:
    """Artifacts for a single Grad-CAM explanation."""

    image_tensor: Tensor  # (3, H, W) normalized tensor
    raw_rgb: np.ndarray  # (H, W, 3) in [0, 1]
    heatmap: np.ndarray  # (H, W) in [0, 1]
    true_label: int
    true_name: str
    pred_label: int
    pred_name: str
    confidence: float
    is_correct: bool


class GradCAM:
    """Compute Gradient-weighted Class Activation Mapping for a given target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = (
            target_layer if target_layer is not None else getattr(model, "gradcam_target_layer")
        )
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, inp: Any, out: Tensor) -> None:
            self.activations = out.detach()

        def backward_hook(module: nn.Module, grad_in: Any, grad_out: tuple[Tensor, ...]) -> None:
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate_heatmap(
        self,
        image: Tensor,  # (1, 3, H, W)
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Compute Grad-CAM heatmap for a single image tensor."""
        self.model.eval()
        image = image.clone().requires_grad_(True)
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)

        pred_class = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_class].item())

        class_to_explain = pred_class if target_class is None else target_class
        score = logits[0, class_to_explain]

        self.model.zero_grad()
        score.backward()

        # Global average pooling of gradients
        # activations: (1, C, H, W), gradients: (1, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to image size
        cam = F.interpolate(
            cam, size=image.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence


def denormalize_image(
    tensor: Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Convert normalized CHW tensor back to HWC numpy float array in [0, 1]."""
    img = tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return img


def collect_gradcam_samples(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    class_names: Sequence[str] | None = None,
    max_correct: int = 3,
    max_wrong: int = 3,
) -> list[GradCAMResult]:
    """Find and compute Grad-CAM for both correct and wrong validation predictions."""
    cam = GradCAM(model)
    resolved = torch.device(device)
    model.to(resolved)
    model.eval()

    correct_samples: list[GradCAMResult] = []
    wrong_samples: list[GradCAMResult] = []

    try:
        for batch in loader:
            images = batch["image"]
            labels = batch["label"]
            for i in range(images.size(0)):
                if len(correct_samples) >= max_correct and len(wrong_samples) >= max_wrong:
                    break

                img_tensor = images[i : i + 1].to(resolved)
                true_lbl = int(labels[i].item())
                heatmap, pred_lbl, conf = cam.generate_heatmap(img_tensor)

                is_correct = (pred_lbl == true_lbl)
                if is_correct and len(correct_samples) >= max_correct:
                    continue
                if not is_correct and len(wrong_samples) >= max_wrong:
                    continue

                true_name = class_names[true_lbl] if class_names else f"Class {true_lbl}"
                pred_name = class_names[pred_lbl] if class_names else f"Class {pred_lbl}"
                raw_rgb = denormalize_image(images[i])

                result = GradCAMResult(
                    image_tensor=images[i],
                    raw_rgb=raw_rgb,
                    heatmap=heatmap,
                    true_label=true_lbl,
                    true_name=true_name,
                    pred_label=pred_lbl,
                    pred_name=pred_name,
                    confidence=conf,
                    is_correct=is_correct,
                )

                if is_correct:
                    correct_samples.append(result)
                else:
                    wrong_samples.append(result)
            if len(correct_samples) >= max_correct and len(wrong_samples) >= max_wrong:
                break
    finally:
        cam.remove_hooks()

    return correct_samples + wrong_samples


def plot_gradcam_grid(
    samples: Sequence[GradCAMResult],
    output_path: str | Path,
    *,
    title: str = "Grad-CAM Attention: Correct vs. Wrong Predictions",
) -> None:
    """Plot a grid comparing original images, heatmaps, and overlays."""
    n_samples = len(samples)
    if n_samples == 0:
        return

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, sample in enumerate(samples):
        # 1. Original Image
        axes[idx, 0].imshow(sample.raw_rgb)
        axes[idx, 0].set_title(f"True: {sample.true_name[:20]}")
        axes[idx, 0].axis("off")

        # 2. Heatmap
        axes[idx, 1].imshow(sample.heatmap, cmap="jet")
        axes[idx, 1].set_title("Neuron Activation (Grad-CAM)")
        axes[idx, 1].axis("off")

        # 3. Overlay
        axes[idx, 2].imshow(sample.raw_rgb)
        axes[idx, 2].imshow(sample.heatmap, cmap="jet", alpha=0.45)
        tag = "CORRECT" if sample.is_correct else "WRONG"
        axes[idx, 2].set_title(
            f"[{tag}] Pred: {sample.pred_name[:18]}\n(Conf: {sample.confidence:.1%})",
            color="darkgreen" if sample.is_correct else "crimson",
            fontweight="bold",
        )
        axes[idx, 2].axis("off")

    plt.suptitle(title, fontsize=14, y=1.002)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
