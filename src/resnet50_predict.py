"""Predict bird species from a local image and a saved ResNet-50 checkpoint.

Uses checkpoint preprocessing and class order. No dataset, pretrained-weight
download, training, classifier refit or confidence recalibration is performed.
"""

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import torch

from src.data.image_io import load_rgb
from src.data.resnet50_transforms import resnet50_transform
from src.models.resnet50 import ResNet50Classifier


def load_classifier(
    checkpoint_path: str | Path, *, device: str = "cpu",
) -> tuple[ResNet50Classifier, dict]:
    """Restore a project ResNet checkpoint and validate inference metadata."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a project ResNet-50 checkpoint mapping.")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its saved configuration.")
    data, model_config = config.get("data"), config.get("model")
    provenance = config.get("data_provenance")
    if not all(isinstance(section, dict) for section in (data, model_config, provenance)):
        raise ValueError("Checkpoint requires data, model and data_provenance metadata.")
    if config.get("architecture", "resnet50") != "resnet50":
        raise ValueError("Only the ordinary ResNet-50 architecture is supported.")
    num_classes = checkpoint.get("num_classes")
    names = provenance.get("class_names")
    if (
        isinstance(num_classes, bool) or not isinstance(num_classes, int)
        or num_classes < 2 or not isinstance(names, list)
        or len(names) != num_classes
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != num_classes
        or model_config.get("num_classes", num_classes) != num_classes
    ):
        raise ValueError("Saved class names must match the classifier's ordered classes.")
    if data.get("normalization") != "imagenet":
        raise ValueError("Unsupported saved normalization; expected 'imagenet'.")
    # Validate required geometry/size before constructing the model. These RGB
    # normalization constants do not load ImageNet pretrained model weights.
    resnet50_transform(
        training=False, image_size=data.get("image_size"), geometry=data.get("geometry"),
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint is missing model_state_dict.")
    model = ResNet50Classifier(
        num_classes=num_classes, weights=None,
        dropout=model_config.get("dropout", 0),
        finetune_scope=model_config.get("finetune_scope", "layer4"),
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, config


@torch.inference_mode()
def predict_image(
    image_path: str | Path, checkpoint_path: str | Path, *,
    device: str = "cpu", top_k: int = 5,
) -> dict:
    """Return ranked species and softmax scores, preserving the saved class IDs."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    image_path = Path(image_path).resolve(strict=True)
    if not image_path.is_file():
        raise ValueError("image_path must be a local image file.")
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    model, config = load_classifier(checkpoint_path, device=device)
    names = config["data_provenance"]["class_names"]
    if top_k > len(names):
        raise ValueError(f"top_k must not exceed the number of classes ({len(names)}).")
    data = config["data"]
    transform = resnet50_transform(
        training=False, image_size=data["image_size"], geometry=data["geometry"],
    )
    # The shared decoder applies EXIF orientation, converts to RGB and fills
    # transparency with white, matching dataset preprocessing.
    with load_rgb({"path": str(image_path)}) as rgb:
        images = transform(rgb).unsqueeze(0).to(device)
    logits = model(images)
    if logits.shape != (1, len(names)) or not torch.isfinite(logits).all():
        raise ValueError("Model returned invalid or nonfinite class logits.")
    scores, indices = logits.softmax(dim=1)[0].topk(top_k)
    return {
        "image": str(image_path), "checkpoint": str(checkpoint_path),
        "device": device,
        "preprocessing": {key: data[key] for key in ("image_size", "geometry", "normalization")},
        "predictions": [
            {"class_id": index, "class_name": names[index], "confidence": score}
            for index, score in zip(indices.cpu().tolist(), scores.cpu().tolist())
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Local bird image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved project model.pt.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are preserved.")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be a positive integer.")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new file.")
    torch.set_num_threads(2)
    try:
        result = predict_image(
            args.image, args.checkpoint, device=args.device, top_k=args.top_k,
        )
        content = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content + "\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
