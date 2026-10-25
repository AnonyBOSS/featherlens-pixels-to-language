

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
import yaml

from src.data.classification import (
    CubClassificationDataset,
    make_classification_loaders,
)
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.transforms import IMAGENET_NORMALIZATION, classification_transform
from src.models.resnet50 import ResNet50Classifier


def _validate_seed(seed: object) -> None:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise ValueError("experiment.seed must be an integer in [0, 2**32).")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read YAML and validate the settings used by experiment setup."""
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    for section in ("experiment", "training", "model", "data"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section {section!r} must be a mapping.")

    name = config["experiment"].get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("experiment.name must be a non-empty string.")

    _validate_seed(config["experiment"].get("seed"))

    if config["training"].get("device") not in ("cpu", "cuda"):
        raise ValueError("training.device must be 'cpu' or 'cuda'.")

    for section, field, minimum in (
        ("model", "num_classes", 2),
        ("data", "image_size", 1),
        ("data", "batch_size", 1),
        ("data", "num_workers", 0),
    ):
        value = config[section].get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{section}.{field} must be an integer >= {minimum}.")

    weights = config["model"].get("weights")
    if not isinstance(weights, str) or weights not in ResNet50_Weights.__members__:
        raise ValueError("model.weights must name pretrained ResNet50_Weights.")

    prepared_dir = config["data"].get("prepared_dir")
    if not isinstance(prepared_dir, str) or not prepared_dir.strip():
        raise ValueError("data.prepared_dir must be a non-empty path string.")
    if config["data"].get("geometry") not in ("fit", "crop"):
        raise ValueError("data.geometry must be 'fit' or 'crop'.")
    if config["data"].get("normalization") != "imagenet":
        raise ValueError("This runner supports data.normalization: imagenet.")

    return config


def seed_everything(seed: int) -> None:
    """Seed random generators and configure deterministic cuDNN behavior."""
    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    """Select CPU or CUDA, requiring CUDA when it is requested."""
    if name not in ("cpu", "cuda"):
        raise ValueError("Device must be 'cpu' or 'cuda'.")

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Run with the CUDA-enabled project environment."
        )

    return torch.device(name)


def prepare_training(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[ResNet50Classifier, dict[str, DataLoader]]:
    """Build train/validation loaders and the pretrained model."""
    data = config["data"]
    prepared_dir = Path(data["prepared_dir"])
    if not prepared_dir.is_absolute():
        prepared_dir = Path(__file__).resolve().parents[2] / prepared_dir

    rows = load_classification_manifest(prepared_dir)
    source = load_cub_split("train")

    if config["model"]["num_classes"] != len(source.class_names):
        raise ValueError("model.num_classes must match the dataset class count.")

    datasets = {
        partition: CubClassificationDataset(
            source,
            rows,
            partition=partition,
            transform=classification_transform(
                training=partition == "train",
                image_size=data["image_size"],
                geometry=data["geometry"],
                normalization=IMAGENET_NORMALIZATION,
            ),
        )
        for partition in ("train", "val")
    }

    loaders = make_classification_loaders(
        datasets,
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        seed=config["experiment"]["seed"],
        pin_memory=device.type == "cuda",
    )

    weights = ResNet50_Weights[config["model"]["weights"]]
    model = ResNet50Classifier(
        num_classes=config["model"]["num_classes"],
        weights=weights,
    ).to(device)

    return model, loaders


@torch.no_grad()
def check_batch(
    model: ResNet50Classifier,
    loader: DataLoader,
    device: torch.device,
    *,
    image_size: int,
) -> dict[str, tuple[int, ...]]:
    """Check one validation batch without changing weights or model mode."""
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("Labels must be a one-dimensional int64 tensor.")
    if (
        images.shape != (labels.shape[0], 3, image_size, image_size)
        or images.dtype != torch.float32
    ):
        raise ValueError("Images must be float32 tensors shaped [B, 3, H, W].")

    was_training = model.training
    model.eval()
    try:
        logits = model(images)
        expected_shape = (labels.shape[0], model.classifier.out_features)
        if logits.shape != expected_shape or not torch.isfinite(logits).all().item():
            raise ValueError("Model outputs have an invalid shape or non-finite values.")
    finally:
        model.train(was_training)

    return {
        "images": tuple(images.shape),
        "labels": tuple(labels.shape),
        "predictions": tuple(logits.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resnet50.yaml"),
        help="YAML configuration path; run from the repository root.",
    )
    parser.add_argument(
        "--check-batch",
        action="store_true",
        help="Check one validation batch without training.",
    )
    args = parser.parse_args()

    batch_shapes = None
    try:
        config = load_config(args.config)
        device = select_device(config["training"]["device"])
        seed_everything(config["experiment"]["seed"])
        print("Preparing train/validation data and pretrained ResNet-50...", flush=True)
        model, loaders = prepare_training(config, device)
        if args.check_batch:
            batch_shapes = check_batch(
                model,
                loaders["val"],
                device,
                image_size=config["data"]["image_size"],
            )
    except (OSError, yaml.YAMLError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Configuration: {args.config.resolve()}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Seed: {config['experiment']['seed']}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    for partition, loader in loaders.items():
        print(f"{partition}: {len(loader.dataset)} images")
    if batch_shapes is not None:
        for name, shape in batch_shapes.items():
            print(f"{name}: {shape}")
        print("Batch check passed. No weights were updated.")
    print("Data and model are ready. Training is not connected yet.")


if __name__ == "__main__":
    main()
