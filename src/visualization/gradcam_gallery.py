"""Generate a curated Grad-CAM visualization gallery for official CUB test predictions.

Produces 8 verified test-set visual explanations using the final M5 checkpoint:
- 3 Confidently correct predictions
- 3 Confidently incorrect predictions (sister-species confusion pairs)
- 2 Borderline / low-confidence predictions

Reuses src.visualization.resnet50_gradcam.gradcam without duplicating mathematical logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from hashlib import file_digest
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn

from src.data.image_io import load_rgb
from src.data.preparation import load_classification_manifest
from src.data.resnet50_transforms import resnet50_transform
from src.data.source import load_cub_split
from src.data.transforms import IMAGENET_NORMALIZATION
from src.resnet50_predict import load_classifier
from src.visualization.resnet50_gradcam import gradcam


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


# Canonical 8 targets derived strictly from M5 test evaluation (07dd204f3ae141e89e626500c142b6a6)
SELECTED_TARGETS = [
    # 1. Confidently Correct (Distinctive species plumage)
    {
        "category": "confident_correct",
        "description": "High-confidence correct: Green Jay with distinctive emerald plumage and facial crest.",
        "test_index": 2102,
    },
    {
        "category": "confident_correct",
        "description": "High-confidence correct: Brown Creeper camouflaged vertically against tree bark.",
        "test_index": 727,
    },
    {
        "category": "confident_correct",
        "description": "High-confidence correct: Green Violetear with striking iridescent hummingbird feathers.",
        "test_index": 1963,
    },
    # 2. Confidently Incorrect (Documented sister-species confusion pairs)
    {
        "category": "confident_incorrect",
        "description": "Confident mistake: Red headed Woodpecker confused with sister species Red bellied Woodpecker.",
        "test_index": 5494,
    },
    {
        "category": "confident_incorrect",
        "description": "Confident mistake: Green Kingfisher confused with Pied Kingfisher due to matching dagger-bill silhouette.",
        "test_index": 2242,
    },
    {
        "category": "confident_incorrect",
        "description": "Confident mistake: Female/immature Summer Tanager confused with Scarlet Tanager (identical olive-yellow plumage).",
        "test_index": 4011,
    },
    # 3. Borderline / Low Confidence (Visual ambiguity / high entropy)
    {
        "category": "borderline",
        "description": "Borderline correct: American Three-toed Woodpecker correctly identified despite heavy visual clutter.",
        "test_index": 5390,
    },
    {
        "category": "borderline",
        "description": "Borderline incorrect: Wilson Warbler in dense vegetation confused with Northern Fulmar.",
        "test_index": 5194,
    },
]


def resolve_gallery_targets(
    evaluation_run_dir: Path,
    manifest_rows: list[dict],
) -> list[GalleryTarget]:
    """Verify target indices against saved test probabilities and manifest metadata."""
    scores_path = evaluation_run_dir / "full_class_scores.npz"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing evaluation scores file: {scores_path}")

    with np.load(scores_path) as data:
        probabilities = data["probabilities"]
        labels = data["labels"]
        image_ids = data["image_ids"].tolist()

    report_path = evaluation_run_dir / "report.json"
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    class_names = report_data["data_provenance"]["class_names"]

    test_manifest = [r for r in manifest_rows if r["partition"] == "test"]
    test_manifest.sort(key=lambda r: r["source_row"])

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
        conf = float(probs[pred_id])
        is_correct = bool(pred_id == true_id)

        target = GalleryTarget(
            category=spec["category"],
            description=spec["description"],
            test_index=idx,
            source_row=int(manifest_row["source_row"]),
            image_id=image_id,
            file_name=manifest_row["file_name"],
            true_class_id=true_id,
            true_class_name=class_names[true_id],
            pred_class_id=pred_id,
            pred_class_name=class_names[pred_id],
            confidence=conf,
            correct=is_correct,
        )
        targets.append(target)

    return targets


def render_gradcam_figure(
    images_tensor: torch.Tensor,
    heatmap: torch.Tensor,
    target: GalleryTarget,
    geometry: str,
    output_png_path: Path,
) -> None:
    """Render a publication-ready 3-panel Grad-CAM figure."""
    mean, std = IMAGENET_NORMALIZATION
    display = images_tensor[0].detach().cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    display = display.clamp(0, 1).permute(1, 2, 0).numpy()
    heatmap_np = heatmap.numpy()

    fig = Figure(figsize=(12, 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # 1. Model input
    axes[0].imshow(display)
    height, width = images_tensor.shape[-2:]
    axes[0].set_title(f"Input Canvas ({width}x{height} fit-and-pad)" if geometry == "fit" else "Model Input")

    # 2. Raw heatmap
    chart = axes[1].imshow(heatmap_np, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Activation Support")
    fig.colorbar(chart, ax=axes[1], shrink=0.75, label="Relative Activation")

    # 3. Overlay
    axes[2].imshow(display)
    axes[2].imshow(heatmap_np, cmap="inferno", vmin=0, vmax=1, alpha=0.65 * heatmap_np)
    axes[2].set_title("Overlay (Network Attention)")

    for ax in axes:
        ax.axis("off")

    status_tag = "CORRECT" if target.correct else "INCORRECT"
    title = f"[{target.category.upper()}] Predicted: {target.pred_class_name} ({target.confidence:.1%}) | True: {target.true_class_name} ({status_tag})"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    with output_png_path.open("wb") as stream:
        fig.savefig(stream, format="png", dpi=160)
    fig.clear()


def generate_gallery(
    checkpoint_path: Path,
    evaluation_run_id: str,
    output_dir: Path,
    device: str = "cuda",
) -> list[dict]:
    """Explain curated images using a checkpoint matching its saved evaluation."""
    checkpoint_path = checkpoint_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    with checkpoint_path.open("rb") as stream:
        checkpoint_sha256 = file_digest(stream, "sha256").hexdigest()

    # Load model and config
    model, config = load_classifier(checkpoint_path, device=device)
    model.eval()

    # Refuse to label one checkpoint's heatmaps with another model's results.
    root = Path(__file__).resolve().parents[2]
    eval_run_dir = root / "outputs/resnet50_inaturalist/evaluation" / evaluation_run_id
    report = json.loads((eval_run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Checkpoint does not match the saved evaluation. Use its matching checkpoint/evaluation run.")
    if report.get("data_provenance") != config["data_provenance"]:
        raise ValueError("Checkpoint and evaluation data provenance/class order differ.")
    preprocessing = {key: config["data"][key] for key in ("image_size", "geometry", "normalization")}
    saved_preprocessing = report.get("preprocessing")
    if not isinstance(saved_preprocessing, dict) or any(
        saved_preprocessing.get(key) != value for key, value in preprocessing.items()
    ):
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

    transform = resnet50_transform(
        training=False,
        image_size=config["data"]["image_size"],
        geometry=config["data"]["geometry"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    gallery_results = []
    print(f"Generating Grad-CAM gallery for {len(targets)} verified test targets...")

    for i, target in enumerate(targets, 1):
        source_item = source.dataset[target.source_row]
        if source_item["label"] != target.true_class_id or source_item["file_name"] != target.file_name:
            raise ValueError(f"Source row {target.source_row} does not match expected target metadata")

        # Save local input PNG for auditability
        local_input_png = inputs_dir / f"test_{target.test_index:04d}_{target.file_name}"
        with load_rgb(source_item["image"]) as pil_image:
            pil_image.save(local_input_png)
            tensor_input = transform(pil_image).unsqueeze(0).to(device)

        # Call existing, verified gradcam function
        target_conv_layer: nn.Module = model.encoder.layer4[-1]
        heatmap, logits, explained_class = gradcam(
            model,
            tensor_input,
            target_conv_layer,
            class_id=None,  # Explain the current model's predicted class.
        )

        probs = logits.softmax(1)[0]
        predicted_id = int(logits.argmax(dim=1).item())
        if predicted_id != target.pred_class_id:
            raise ValueError(f"Current prediction differs from the saved evaluation for {target.image_id}. Regenerate the evaluation before using this curated gallery.")
        # Labels, confidence and correctness must describe this same forward pass.
        target = replace(
            target,
            pred_class_id=predicted_id,
            pred_class_name=class_names[predicted_id],
            confidence=float(probs[predicted_id]),
            correct=predicted_id == target.true_class_id,
        )
        top5_indices = probs.topk(5).indices.cpu().tolist()
        top5_list = [
            {"class_id": idx, "class_name": class_names[idx], "probability": float(probs[idx])}
            for idx in top5_indices
        ]

        file_prefix = f"{i:02d}_{target.category}_idx{target.test_index}"
        png_path = output_dir / f"{file_prefix}.png"
        json_path = output_dir / f"{file_prefix}.json"

        # Render 3-panel figure
        render_gradcam_figure(tensor_input, heatmap, target, config["data"]["geometry"], png_path)

        record = {
            "index": i,
            "category": target.category,
            "description": target.description,
            "test_index": target.test_index,
            "source_row": target.source_row,
            "image_id": target.image_id,
            "file_name": target.file_name,
            "local_input_image": str(local_input_png),
            "output_figure": str(png_path),
            "true_class_id": target.true_class_id,
            "true_class_name": target.true_class_name,
            "predicted_class_id": target.pred_class_id,
            "predicted_class_name": target.pred_class_name,
            "confidence": target.confidence,
            "correct": target.correct,
            "explained_class_id": explained_class,
            "explained_class_name": class_names[explained_class],
            "top5": top5_list,
            "has_spatial_contrast": bool(heatmap.max() > 0),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "evaluation_run_id": evaluation_run_id,
            "target_layer": "encoder.layer4.2",
            "image_size": config["data"]["image_size"],
        }
        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        gallery_results.append(record)
        print(f"[{i}/{len(targets)}] Generated: {png_path.name} | Pred: {target.pred_class_name} ({target.confidence:.1%}) | True: {target.true_class_name}")

    # Write manifest and summary markdown
    manifest_file = output_dir / "gallery_manifest.json"
    manifest_file.write_text(json.dumps(gallery_results, indent=2), encoding="utf-8")

    generate_markdown_summary(gallery_results, output_dir / "README.md")
    print(f"Gallery generation complete! Saved to {output_dir}")
    return gallery_results


def generate_markdown_summary(results: list[dict], summary_path: Path) -> None:
    """Generate Markdown summary table for presentation and documentation."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/resnet50_inaturalist/augmented_pca_search/accuracy_challenger/model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--evaluation-run",
        type=str,
        default="07dd204f3ae141e89e626500c142b6a6",
        help="Run ID of the official test evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gradcam_gallery"),
        help="Output directory for gallery figures and metadata",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    generate_gallery(args.checkpoint, args.evaluation_run, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
