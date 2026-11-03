"""Grad-CAM for one raw component class score, never an ensemble probability.

Maps are relative attributions, not segmentation or proof of causality.
"""
import math
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F


def compute_gradcam(model: nn.Module, images: Tensor, target_class: int | None = None) -> dict:
    """Attribute one FP32 component score; retain weights, grads and training flags.

    Pass one preprocessed NCHW image on the model's device. Frozen parameters
    work because a cloned input enables autograd. No optimizer or backward()
    call is used. Constant/ReLU-zero maps return explicit degenerate zero maps.
    """
    if (not isinstance(images, Tensor) or images.ndim != 4 or images.shape[:2] != (1, 3)
            or min(images.shape[-2:]) < 1 or not images.is_floating_point()):
        raise ValueError("images must be one floating-point NCHW RGB image")
    if not torch.isfinite(images).all():
        raise ValueError("Nonfinite input image")
    if target_class is not None and (type(target_class) is not int or not 0 <= target_class < 200):
        raise ValueError("target_class must be an integer in [0, 199] or None")
    captured = []
    states = [(module, module.training) for module in model.modules()]
    hook = model.gradcam_target_layer.register_forward_hook(
        lambda module, inputs, output: captured.append(output))
    try:
        model.eval()
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
                device_type=images.device.type, enabled=False):
            image = images.detach().to(dtype=torch.float32).clone().requires_grad_(True)
            logits = model(image)
            if (not isinstance(logits, Tensor) or logits.shape != (1, 200)
                    or logits.dtype != torch.float32 or not torch.isfinite(logits).all()):
                raise ValueError("Nonfinite or non-FP32 component logits; expected shape [1, 200]")
            if (len(captured) != 1 or not isinstance(captured[0], Tensor)
                    or captured[0].ndim != 4 or captured[0].shape[0] != 1
                    or not captured[0].requires_grad):
                raise ValueError("Target layer must capture one tracked spatial NCHW activation")
            activation = captured[0]
            predicted = int(logits[0].argmax())
            target = predicted if target_class is None else target_class
            gradient, = torch.autograd.grad(logits[0, target], activation)
            if (activation.dtype != torch.float32 or gradient.dtype != torch.float32
                    or not torch.isfinite(activation).all() or not torch.isfinite(gradient).all()):
                raise ValueError("Nonfinite or non-FP32 activation or raw component-score gradient")
            raw = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            if not torch.isfinite(raw).all():
                raise ValueError("Nonfinite raw Grad-CAM")
            low, high = raw.min().item(), raw.max().item()
            degenerate = high - low <= 1e-12
            normalized = torch.zeros_like(raw) if degenerate else (raw - low) / (high - low)
            heat = F.interpolate(normalized, size=image.shape[-2:],
                                 mode="bilinear", align_corners=False)[0, 0]
            if not torch.isfinite(heat).all():
                raise ValueError("Nonfinite upsampled Grad-CAM")
            return {
                "heatmap": heat.detach().cpu().float(),
                "logits": logits[0].detach().cpu().float(),
                "target_class": target, "predicted_class": predicted,
                "target_component_logit": logits[0, target].item(),
                "layer": model.spec.gradcam_target_path,
                "activation_shape": list(activation.shape), "gradient_shape": list(gradient.shape),
                "finite": True, "degenerate": degenerate, "raw_cam_min": low, "raw_cam_max": high,
                "precision": "FP32 without autocast",
                "attribution": "raw component class score",
            }
    finally:
        hook.remove()
        for module, training in states:
            module.training = training


def render_gradcam(
    image: Image.Image, heatmap: Tensor, image_size: int, *,
    output_path: str | Path, alpha: float = 0.4,
) -> Path:
    """Save a full native-image overlay after removing centered fit padding.

    image is the EXIF-corrected RGB source used by preprocessing. heatmap is
    the square input-canvas map returned by compute_gradcam. Colors use each
    map's relative [0, 1] scale; they are not absolute component contributions.
    """
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("image must be an EXIF-corrected RGB PIL image")
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    heat = torch.as_tensor(heatmap).detach().cpu().float()
    if (heat.shape != (image_size, image_size) or not torch.isfinite(heat).all()
            or heat.min() < 0 or heat.max() > 1):
        raise ValueError("heatmap must be a finite [0, 1] input-canvas map")
    # Exact geometry from data.transforms._FitAndPad, including half-up rounding.
    scale = image_size / max(image.size)
    width = max(1, min(image_size, math.floor(image.width * scale + 0.5)))
    height = max(1, min(image_size, math.floor(image.height * scale + 0.5)))
    left, top = (image_size - width) // 2, (image_size - height) // 2
    with Image.fromarray(heat.numpy()) as canvas:
        with canvas.crop((left, top, left + width, top + height)) as content:
            native = content.resize(image.size, resample=Image.Resampling.BILINEAR)
    try:
        import numpy as np
        from matplotlib import colormaps

        colors = (colormaps["jet"](np.asarray(native))[:, :, :3] * 255).round().astype("uint8")
        with Image.fromarray(colors) as colored:
            with Image.blend(image, colored, float(alpha)) as overlay:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                overlay.save(path)
        return path
    finally:
        native.close()
