"""Class-specific Grad-CAM for saved ResNet-50 classifiers, without training.

Method: Selvaraju et al., https://arxiv.org/abs/1610.02391.
Heatmaps use logits, average spatial gradients and positive weighted activations.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.data.image_io import load_rgb
from src.data.resnet50_transforms import resnet50_transform
from src.data.transforms import IMAGENET_NORMALIZATION
from src.resnet50_predict import load_classifier


def gradcam(
    model: nn.Module, images: Tensor, target_layer: nn.Module, *,
    class_id: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Return an input-sized CPU heatmap, logits and the explained class ID.

    The caller supplies an evaluation-mode model and a target layer used once.
    Frozen parameters are supported by enabling gradients on a private input
    copy. Parameter .grad values and model state are not modified. A zero map
    means no positive spatial contrast was found; it is not an error.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("Grad-CAM requires autograd; call outside torch.inference_mode().")
    if any(module.training for module in model.modules()):
        raise ValueError("Grad-CAM requires model.eval() to preserve evaluation behavior.")
    if (
        images.ndim != 4 or images.shape[0] != 1 or images.shape[1] != 3
        or min(images.shape[2:]) < 1 or not images.is_floating_point()
        or not torch.isfinite(images).all()
    ):
        raise ValueError("Expected one finite floating-point RGB image [1, 3, H, W].")
    if class_id is not None and (isinstance(class_id, bool) or not isinstance(class_id, int) or class_id < 0):
        raise ValueError("class_id must be a nonnegative integer or None.")
    activations = []

    def capture(_module, _inputs, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            logits = model(images.detach().clone().requires_grad_(True))
            if (
                logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] < 2
                or not torch.isfinite(logits).all()
            ):
                raise ValueError("Expected finite class logits shaped [1, classes].")
            if len(activations) != 1 or not isinstance(activations[0], Tensor):
                raise ValueError("Target layer must produce a spatial tensor exactly once.")
            features = activations[0]
            if features.ndim != 4 or features.shape[0] != 1 or not features.requires_grad:
                raise ValueError("Target layer must expose differentiable spatial features.")
            selected = int(logits.argmax(1).item()) if class_id is None else class_id
            if selected >= logits.shape[1]:
                raise ValueError("class_id is outside the saved classifier's class range.")
            gradients = torch.autograd.grad(logits[0, selected], features)[0]
            if not torch.isfinite(features).all() or not torch.isfinite(gradients).all():
                raise ValueError("Nonfinite Grad-CAM features or gradients.")
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            heatmap = F.relu((weights * features).sum(dim=1, keepdim=True))
            heatmap = F.interpolate(heatmap, size=images.shape[2:], mode="bilinear", align_corners=False)[0, 0]
            heatmap = heatmap - heatmap.min()
            peak = heatmap.max()
            if peak > 0:
                heatmap = heatmap / peak
            return heatmap.detach().cpu(), logits.detach().cpu(), selected
    finally:
        # Also remove the hook if a malformed input/model raises an exception.
        handle.remove()


def explain_image(
    image_path: str | Path, checkpoint_path: str | Path, output_path: str | Path, *,
    device: str = "cpu", class_id: int | None = None, true_class_id: int | None = None,
) -> dict:
    """Save a three-panel PNG and adjacent JSON report without overwriting files."""
    output_path = Path(output_path).resolve()
    report_path = output_path.with_suffix(".json")
    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png.")
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Output PNG or JSON already exists; choose a new name.")
    image_path = Path(image_path).resolve(strict=True)
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    if not image_path.is_file():
        raise ValueError("image_path must be a local image file.")
    model, config = load_classifier(checkpoint_path, device=device)
    names = config["data_provenance"]["class_names"]
    if true_class_id is not None and (
        isinstance(true_class_id, bool) or not isinstance(true_class_id, int)
        or not 0 <= true_class_id < len(names)
    ):
        raise ValueError("true_class_id must be a saved zero-based class ID.")
    data = config["data"]
    transform = resnet50_transform(training=False, image_size=data["image_size"], geometry=data["geometry"])
    with load_rgb({"path": str(image_path)}) as rgb:
        images = transform(rgb).unsqueeze(0).to(device)
    heatmap, logits, selected = gradcam(model, images, model.encoder.layer4[-1], class_id=class_id)
    probabilities = logits.softmax(1)[0]
    predicted = int(probabilities.argmax().item())
    report = {
        "method": "Grad-CAM", "method_source": "https://arxiv.org/abs/1610.02391",
        "image": str(image_path), "checkpoint": str(checkpoint_path), "device": device,
        "target_layer": "encoder.layer4.2", "gradient_target": "raw_class_logit",
        "preprocessing": {key: data[key] for key in ("image_size", "geometry", "normalization")},
        "predicted_class_id": predicted, "predicted_class_name": names[predicted],
        "confidence": float(probabilities[predicted]),
        "explained_class_id": selected, "explained_class_name": names[selected],
        "true_class_id": true_class_id,
        "true_class_name": None if true_class_id is None else names[true_class_id],
        "correct": None if true_class_id is None else predicted == true_class_id,
        "has_spatial_contrast": bool(heatmap.max() > 0),
        "weights_updated": False, "png": str(output_path),
    }
    # Display the exact network input canvas, so padding and heatmaps align.
    mean, std = IMAGENET_NORMALIZATION
    display = images[0].detach().cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    display = display.clamp(0, 1).permute(1, 2, 0).numpy()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    axes[0].imshow(display)
    axes[0].set_title("Model input (fit-and-pad)" if data["geometry"] == "fit" else "Model input (crop)")
    chart = axes[1].imshow(heatmap.numpy(), cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM: positive class support")
    axes[2].imshow(display)
    axes[2].imshow(heatmap.numpy(), cmap="inferno", vmin=0, vmax=1, alpha=0.65 * heatmap.numpy())
    axes[2].set_title("Overlay on model input")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(chart, ax=axes[1], shrink=0.7, label="Relative activation")
    title = f"Predicted: {names[predicted]} ({probabilities[predicted]:.1%}) | Explained: {names[selected]}"
    if true_class_id is not None:
        title += f"\nTrue: {names[true_class_id]} | {'Correct' if report['correct'] else 'Incorrect'} prediction"
    if not report["has_spatial_contrast"]:
        title += "\nNo positive spatial contrast found"
    figure.suptitle(title, fontsize=11)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=140)
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    finally:
        figure.clear()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New PNG path; also saves adjacent JSON.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--class-id", type=int, help="Explain this zero-based class ID; default: predicted class.")
    parser.add_argument("--true-class-id", type=int, help="Known label for marking correct/incorrect predictions.")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    try:
        report = explain_image(
            args.image, args.checkpoint, args.output, device=args.device,
            class_id=args.class_id, true_class_id=args.true_class_id,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Grad-CAM failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
