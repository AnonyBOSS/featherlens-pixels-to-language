

import argparse
import math
import random
import re
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
from src.data.resnet50_transforms import resnet50_transform
from src.models.resnet50 import (
    INATURALIST_WEIGHTS, ResNet50Classifier, load_inaturalist_encoder,
)
from src.training.resnet50_experiment import run_experiment


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

    for section in ("experiment", "training", "model", "data", "tracking", "checkpoints"):
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
    if weights == INATURALIST_WEIGHTS:
        path = config["model"].get("pretrained_checkpoint")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("model.pretrained_checkpoint must be a non-empty path string.")
        digest = config["model"].get("pretrained_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("model.pretrained_sha256 must contain 64 hexadecimal characters.")
    elif not isinstance(weights, str) or weights not in ResNet50_Weights.__members__:
        raise ValueError("model.weights must name ResNet50_Weights or INAT2021_FROM_SCRATCH.")
    if config["model"].get("finetune_scope", "layer4") not in ("layer4", "last_block"):
        raise ValueError("model.finetune_scope must be 'layer4' or 'last_block'.")

    prepared_dir = config["data"].get("prepared_dir")
    if not isinstance(prepared_dir, str) or not prepared_dir.strip():
        raise ValueError("data.prepared_dir must be a non-empty path string.")
    if config["data"].get("geometry") not in ("fit", "crop"):
        raise ValueError("data.geometry must be 'fit' or 'crop'.")
    if config["data"].get("normalization") != "imagenet":
        raise ValueError("This runner supports data.normalization: imagenet.")
    if config["data"].get("augmentation", "baseline") not in ("baseline", "mild", "spatial"):
        raise ValueError("data.augmentation must be 'baseline', 'mild' or 'spatial'.")
    if config["data"].get("augmentation") == "spatial" and config["data"]["geometry"] != "fit":
        raise ValueError("Spatial augmentation requires data.geometry: fit.")
    for name, value in (
        ("model.dropout", config["model"].get("dropout", 0.0)),
        ("training.label_smoothing", config["training"].get("label_smoothing", 0.0)),
    ):
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value < 1
        ):
            raise ValueError(f"{name} must be a finite probability in [0, 1).")

    if config["training"].get("optimizer") != "adamw":
        raise ValueError("training.optimizer must be 'adamw'.")
    if config["tracking"].get("backend") != "mlflow":
        raise ValueError("tracking.backend must be 'mlflow'.")
    if config["checkpoints"].get("selection_metric") != "val_accuracy":
        raise ValueError("checkpoints.selection_metric must be 'val_accuracy'.")
    for section, field in (("tracking", "local_dir"), ("checkpoints", "local_root")):
        value = config[section].get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{field} must be a non-empty path string.")

    rates = {"training.weight_decay": config["training"].get("weight_decay")}
    for stage, fields in (
        ("head", ("learning_rate",)),
        ("finetune", ("classifier_learning_rate", "encoder_learning_rate")),
    ):
        settings = config["training"].get(stage)
        if not isinstance(settings, dict):
            raise ValueError(f"training.{stage} must be a mapping.")
        epochs = settings.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"training.{stage}.epochs must be a positive integer.")
        smoothing = settings.get("label_smoothing", config["training"].get("label_smoothing", 0.0))
        if (
            isinstance(smoothing, bool) or not isinstance(smoothing, (int, float))
            or not 0 <= smoothing < 1
        ):
            raise ValueError(f"training.{stage}.label_smoothing must be in [0, 1).")
        mixup = settings.get("mixup_alpha", 0.0)
        if (
            isinstance(mixup, bool) or not isinstance(mixup, (int, float))
            or not math.isfinite(mixup) or mixup < 0
        ):
            raise ValueError(f"training.{stage}.mixup_alpha must be finite and non-negative.")
        for field in fields:
            rates[f"training.{stage}.{field}"] = settings.get(field)
    for name, value in rates.items():
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (value == 0 and name != "training.weight_decay")
        ):
            raise ValueError(f"{name} must be finite and positive (weight decay may be zero).")

    stopping = config["training"]["finetune"].get("early_stopping")
    if stopping is not None:
        if not isinstance(stopping, dict):
            raise ValueError("training.finetune.early_stopping must be a mapping.")
        patience = stopping.get("patience")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("early_stopping.patience must be a positive integer.")
        delta = stopping.get("min_delta", 0.0)
        if (
            isinstance(delta, bool) or not isinstance(delta, (int, float))
            or not math.isfinite(delta) or delta < 0
        ):
            raise ValueError("early_stopping.min_delta must be finite and non-negative.")

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
            transform=resnet50_transform(
                training=partition == "train",
                image_size=data["image_size"],
                geometry=data["geometry"],
                augmentation=data.get("augmentation", "baseline"),
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

    use_inaturalist = config["model"]["weights"] == INATURALIST_WEIGHTS
    weights = None if use_inaturalist else ResNet50_Weights[config["model"]["weights"]]
    model = ResNet50Classifier(
        num_classes=config["model"]["num_classes"],
        weights=weights,
        dropout=config["model"].get("dropout", 0.0),
        finetune_scope=config["model"].get("finetune_scope", "layer4"),
    )
    if use_inaturalist:
        checkpoint_path = Path(config["model"]["pretrained_checkpoint"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
        load_inaturalist_encoder(model, checkpoint_path, config["model"]["pretrained_sha256"])
    model = model.to(device)

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
        default=Path("configs/resnet50_inaturalist.yaml"),
        help="YAML configuration path; run from the repository root.",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--train",
        action="store_true",
        help="Run the full two-stage training experiment from the configuration.",
    )
    modes.add_argument(
        "--check-batch",
        action="store_true",
        help="Check one validation batch without training.",
    )
    modes.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train one epoch per stage on at most eight images per partition.",
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
        else:
            run_experiment(model, loaders, config, device, smoke_test=args.smoke_test)
    except KeyboardInterrupt:
        parser.exit(130, "Training interrupted; completed-epoch checkpoints were kept.\n")
    except (OSError, yaml.YAMLError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Configuration: {args.config.resolve()}")
    experiment_name = config["experiment"]["name"] + ("_smoke" if args.smoke_test else "")
    print(f"Experiment: {experiment_name}")
    print(f"Seed: {config['experiment']['seed']}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    for partition, loader in loaders.items():
        print(f"Available {partition}: {len(loader.dataset)} images")
    if batch_shapes is not None:
        for name, shape in batch_shapes.items():
            print(f"{name}: {shape}")
        print("Batch check passed. No weights were updated.")
    else:
        print("Experiment completed. See its printed checkpoint directory and MLflow run.")
        if args.smoke_test:
            print("Smoke mode used at most eight images per partition; scores are not evaluation results.")


if __name__ == "__main__":
    main()
