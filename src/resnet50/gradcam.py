"""Grad-CAM heatmaps, single-image overlays, and curated test gallery for ResNet-50."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import file_digest
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.data.image_io import load_rgb
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.transforms import IMAGENET_NORMALIZATION
from src.resnet50.model import load_classifier, resnet50_transform


def gradcam(
    model: nn.Module, images: Tensor, target_layer: nn.Module, *,
    class_id: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Compute a normalized [0, 1] Grad-CAM heatmap, logits, and target class id."""
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
        handle.remove()


def explain_image(
    image_path: str | Path, checkpoint_path: str | Path, output_path: str | Path, *,
    device: str = "cpu", class_id: int | None = None, true_class_id: int | None = None,
) -> dict:
    """Save a 3-panel Grad-CAM PNG and adjacent JSON report for one local image."""
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
    mean, std = IMAGENET_NORMALIZATION
    display = images[0].detach().cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    display = display.clamp(0, 1).permute(1, 2, 0).numpy()

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


@dataclass(frozen=True)
class GalleryTarget:
    category: str
    description: str
    test_index: int
    source_row: int
    image_id: str
    file_name: str
    true_class_id: int
    true_class_name: str
    pred_class_id: int
    pred_class_name: str
    confidence: float
    correct: bool


SELECTED_TARGETS = [
    {"category": "confident_correct", "description": "High-confidence correct: Green Jay with distinctive emerald plumage and facial crest.", "test_index": 2102},
    {"category": "confident_correct", "description": "High-confidence correct: Brown Creeper camouflaged vertically against tree bark.", "test_index": 727},
    {"category": "confident_correct", "description": "High-confidence correct: Green Violetear with striking iridescent hummingbird feathers.", "test_index": 1963},
    {"category": "confident_incorrect", "description": "Confident mistake: Red headed Woodpecker confused with sister species Red bellied Woodpecker.", "test_index": 5494},
    {"category": "confident_incorrect", "description": "Confident mistake: Green Kingfisher confused with Pied Kingfisher due to matching dagger-bill silhouette.", "test_index": 2242},
    {"category": "confident_incorrect", "description": "Confident mistake: Female/immature Summer Tanager confused with Scarlet Tanager (identical olive-yellow plumage).", "test_index": 4011},
    {"category": "borderline", "description": "Borderline correct: American Three-toed Woodpecker correctly identified despite heavy visual clutter.", "test_index": 5390},
    {"category": "borderline", "description": "Borderline incorrect: Wilson Warbler in dense vegetation confused with Northern Fulmar.", "test_index": 5194},
]


def resolve_gallery_targets(evaluation_run_dir: Path, manifest_rows: list[dict]) -> list[GalleryTarget]:
    """Load target test predictions and labels from a saved evaluation run."""
    scores_path = evaluation_run_dir / "full_class_scores.npz"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing evaluation scores file: {scores_path}")
    with np.load(scores_path) as data:
        probabilities, labels, image_ids = data["probabilities"], data["labels"], data["image_ids"].tolist()
    class_names = json.loads((evaluation_run_dir / "report.json").read_text(encoding="utf-8"))["data_provenance"]["class_names"]
    test_manifest = sorted((r for r in manifest_rows if r["partition"] == "test"), key=lambda r: r["source_row"])
    targets: list[GalleryTarget] = []
    for spec in SELECTED_TARGETS:
        idx = spec["test_index"]
        manifest_row = test_manifest[idx]
        image_id = image_ids[idx]
        if image_id != manifest_row["image_id"]:
            raise ValueError(f"Image ID mismatch at test index {idx}: {image_id} != {manifest_row['image_id']}")
        true_id = int(labels[idx])
        if true_id != manifest_row["class_id"]:
            raise ValueError(f"True label mismatch at test index {idx}.")
        probs = probabilities[idx]
        pred_id = int(probs.argmax())
        targets.append(GalleryTarget(
            category=spec["category"], description=spec["description"], test_index=idx,
            source_row=int(manifest_row["source_row"]), image_id=image_id, file_name=manifest_row["file_name"],
            true_class_id=true_id, true_class_name=class_names[true_id],
            pred_class_id=pred_id, pred_class_name=class_names[pred_id],
            confidence=float(probs[pred_id]), correct=bool(pred_id == true_id),
        ))
    return targets


def render_gradcam_figure(
    images_tensor: torch.Tensor, heatmap: torch.Tensor, target: GalleryTarget,
    geometry: str, output_png_path: Path,
) -> None:
    """Save a 3-panel Grad-CAM figure for one gallery target."""
    mean, std = IMAGENET_NORMALIZATION
    display = images_tensor[0].detach().cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    display = display.clamp(0, 1).permute(1, 2, 0).numpy()
    heatmap_np = heatmap.numpy()
    fig = Figure(figsize=(12, 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    axes[0].imshow(display)
    height, width = images_tensor.shape[-2:]
    axes[0].set_title(f"Input Canvas ({width}x{height} fit-and-pad)" if geometry == "fit" else "Model Input")
    chart = axes[1].imshow(heatmap_np, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Activation Support")
    fig.colorbar(chart, ax=axes[1], shrink=0.75, label="Relative Activation")
    axes[2].imshow(display)
    axes[2].imshow(heatmap_np, cmap="inferno", vmin=0, vmax=1, alpha=0.65 * heatmap_np)
    axes[2].set_title("Overlay (Network Attention)")
    for ax in axes:
        ax.axis("off")
    status_tag = "CORRECT" if target.correct else "INCORRECT"
    fig.suptitle(f"[{target.category.upper()}] Predicted: {target.pred_class_name} ({target.confidence:.1%}) | True: {target.true_class_name} ({status_tag})", fontsize=11, fontweight="bold")
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    with output_png_path.open("wb") as stream:
        fig.savefig(stream, format="png", dpi=160)
    fig.clear()


def generate_gallery(
    checkpoint_path: Path, evaluation_run_id: str, output_dir: Path, device: str = "cuda",
) -> list[dict]:
    """Generate the 8-image official-test Grad-CAM gallery matched to an evaluation run."""
    checkpoint_path = checkpoint_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    with checkpoint_path.open("rb") as stream:
        checkpoint_sha256 = file_digest(stream, "sha256").hexdigest()
    model, config = load_classifier(checkpoint_path, device=device)
    model.eval()
    root = Path(__file__).resolve().parents[2]
    eval_run_dir = root / "outputs/resnet50_inaturalist/evaluation" / evaluation_run_id
    report = json.loads((eval_run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Checkpoint does not match the saved evaluation. Use its matching checkpoint/evaluation run.")
    if report.get("data_provenance") != config["data_provenance"]:
        raise ValueError("Checkpoint and evaluation data provenance/class order differ.")
    preprocessing = {key: config["data"][key] for key in ("image_size", "geometry", "normalization")}
    saved_preprocessing = report.get("preprocessing")
    if not isinstance(saved_preprocessing, dict) or any(saved_preprocessing.get(key) != value for key, value in preprocessing.items()):
        raise ValueError("Checkpoint and evaluation preprocessing differ.")
    prepared_dir = Path(config["data"]["prepared_dir"])
    if not prepared_dir.is_absolute():
        prepared_dir = root / prepared_dir
    manifest_rows = load_classification_manifest(prepared_dir)
    targets = resolve_gallery_targets(eval_run_dir, manifest_rows)
    source = load_cub_split("test")
    class_names = config["data_provenance"]["class_names"]
    if list(source.class_names) != class_names:
        raise ValueError("Test source class order differs from the checkpoint.")
    transform = resnet50_transform(training=False, image_size=config["data"]["image_size"], geometry=config["data"]["geometry"])
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    gallery_results = []
    print(f"Generating Grad-CAM gallery for {len(targets)} verified test targets...")
    for i, target in enumerate(targets, 1):
        source_item = source.dataset[target.source_row]
        if source_item["label"] != target.true_class_id or source_item["file_name"] != target.file_name:
            raise ValueError(f"Source row {target.source_row} does not match expected target metadata")
        local_input_png = inputs_dir / f"test_{target.test_index:04d}_{target.file_name}"
        with load_rgb(source_item["image"]) as pil_image:
            pil_image.save(local_input_png)
            tensor_input = transform(pil_image).unsqueeze(0).to(device)
        target_conv_layer: nn.Module = model.encoder.layer4[-1]
        heatmap, logits, explained_class = gradcam(model, tensor_input, target_conv_layer, class_id=None)
        probs = logits.softmax(1)[0]
        predicted_id = int(logits.argmax(dim=1).item())
        if predicted_id != target.pred_class_id:
            raise ValueError(f"Current prediction differs from the saved evaluation for {target.image_id}. Regenerate the evaluation before using this curated gallery.")
        target = replace(
            target, pred_class_id=predicted_id, pred_class_name=class_names[predicted_id],
            confidence=float(probs[predicted_id]), correct=predicted_id == target.true_class_id,
        )
        top5_list = [{"class_id": idx, "class_name": class_names[idx], "probability": float(probs[idx])} for idx in probs.topk(5).indices.cpu().tolist()]
        file_prefix = f"{i:02d}_{target.category}_idx{target.test_index}"
        png_path, json_path = output_dir / f"{file_prefix}.png", output_dir / f"{file_prefix}.json"
        render_gradcam_figure(tensor_input, heatmap, target, config["data"]["geometry"], png_path)
        record = {
            "index": i, "category": target.category, "description": target.description,
            "test_index": target.test_index, "source_row": target.source_row,
            "image_id": target.image_id, "file_name": target.file_name,
            "local_input_image": str(local_input_png), "output_figure": str(png_path),
            "true_class_id": target.true_class_id, "true_class_name": target.true_class_name,
            "predicted_class_id": target.pred_class_id, "predicted_class_name": target.pred_class_name,
            "confidence": target.confidence, "correct": target.correct,
            "explained_class_id": explained_class, "explained_class_name": class_names[explained_class],
            "top5": top5_list, "has_spatial_contrast": bool(heatmap.max() > 0),
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
            "evaluation_run_id": evaluation_run_id, "target_layer": "encoder.layer4.2",
            "image_size": config["data"]["image_size"],
        }
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        gallery_results.append(record)
        print(f"[{i}/{len(targets)}] Generated: {png_path.name} | Pred: {target.pred_class_name} ({target.confidence:.1%}) | True: {target.true_class_name}")
    (output_dir / "gallery_manifest.json").write_text(json.dumps(gallery_results, indent=2), encoding="utf-8")
    generate_markdown_summary(gallery_results, output_dir / "README.md")
    print(f"Gallery generation complete! Saved to {output_dir}")
    return gallery_results


def generate_markdown_summary(results: list[dict], summary_path: Path) -> None:
    lines = [
        "# Official CUB Test Split Grad-CAM Gallery",
        "",
        "These official-test explanations use a checkpoint matched to its saved evaluation. Labels and confidence are calculated from the same forward pass as each heatmap.",
        "",
        "| # | Category | True Species | Predicted Species | Confidence | Status | Figure |",
        "|---|---|---|---|---:|:---:|---|",
    ]
    for r in results:
        status_icon = "PASS" if r["correct"] else "FAIL"
        rel_png = Path(r["output_figure"]).name
        lines.append(
            f"| {r['index']} | `{r['category']}` | {r['true_class_name']} | {r['predicted_class_name']} | {r['confidence']:.2%} | {status_icon} | [{rel_png}]({rel_png}) |"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def gallery_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate curated Grad-CAM test gallery.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt"))
    parser.add_argument("--evaluation-run", type=str, default="07dd204f3ae141e89e626500c142b6a6")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gradcam_gallery"))
    parser.add_argument("--device", type=str, choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    generate_gallery(args.checkpoint, args.evaluation_run, args.output_dir, device=args.device)
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "gallery":
        return gallery_main(raw[1:])
    if "--gallery" in raw:
        raw = [arg for arg in raw if arg != "--gallery"]
        return gallery_main(raw)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New PNG path; also saves adjacent JSON.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--class-id", type=int, help="Explain this zero-based class ID; default: predicted class.")
    parser.add_argument("--true-class-id", type=int, help="Known label for marking correct/incorrect predictions.")
    args = parser.parse_args(raw)
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
