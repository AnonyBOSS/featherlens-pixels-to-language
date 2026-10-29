"""Refit a regularized linear classifier on frozen ResNet-50 features.

Feature statistics are learned on training data only. Optional positive logit
scaling uses validation labels for calibration, so validation NLL is a tuning
measurement, not an independent test estimate. No test split is loaded here.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from hashlib import file_digest
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class HeadFit:
    weight: Tensor
    bias: Tensor
    logit_scale: float
    train_metrics: dict[str, float | int]
    val_metrics: dict[str, float | int]


def _validate_features(
    train_features: Tensor, train_labels: Tensor,
    val_features: Tensor, val_labels: Tensor, num_classes: int,
) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    for name, features, labels in (
        ("train", train_features, train_labels), ("val", val_features, val_labels),
    ):
        if (
            features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1
            or not features.is_floating_point() or not torch.isfinite(features).all().item()
            or labels.ndim != 1 or labels.shape[0] != features.shape[0]
            or labels.dtype != torch.int64
            or labels.min().item() < 0 or labels.max().item() >= num_classes
        ):
            raise ValueError(f"Invalid {name} features or labels.")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Training and validation feature dimensions must match.")
    if train_labels.unique().numel() != num_classes:
        raise ValueError("Every class must appear in the training labels.")


def _validate_shrinkage(shrinkage: float) -> None:
    if (
        isinstance(shrinkage, bool) or not isinstance(shrinkage, (int, float))
        or not math.isfinite(shrinkage) or not 0 < shrinkage <= 1
    ):
        raise ValueError("shrinkage must be finite and in (0, 1].")


def classification_metrics(logits: Tensor, labels: Tensor) -> dict[str, float | int]:
    """Compute unsmoothed NLL, accuracy and unweighted per-class mean F1."""
    logits, labels = logits.detach().cpu(), labels.detach().cpu()
    predictions = logits.argmax(1)
    classes = logits.shape[1]
    matrix = torch.bincount(
        labels * classes + predictions, minlength=classes * classes,
    ).reshape(classes, classes).double()
    f1 = 2 * matrix.diag() / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    return {
        "samples": labels.numel(),
        "accuracy": (predictions == labels).double().mean().item(),
        "loss": nn.functional.cross_entropy(logits.double(), labels).item(),
        "macro_f1": f1.mean().item(),
    }


def calibrate_logit_scale(logits: Tensor, labels: Tensor) -> float:
    """Fit one positive scale to validation NLL without changing class ordering."""
    scores, labels = logits.detach().cpu().double(), labels.detach().cpu()
    target_mean = scores.gather(1, labels[:, None]).mean().item()

    def derivative(scale: float) -> float:
        return (scores * (scores * scale).softmax(1)).sum(1).mean().item() - target_mean

    if derivative(0.0) >= 0:
        return 1e-6
    lower, upper = 0.0, 1.0
    while derivative(upper) < 0 and upper < 1e7:
        upper *= 2
    for _ in range(35):
        middle = (lower + upper) / 2
        if derivative(middle) < 0:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def fold_feature_standardization(
    weight: Tensor, bias: Tensor, mean: Tensor, std: Tensor, scale: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Absorb train-only feature normalization into a plain linear layer."""
    mean, std = mean.detach().cpu().double(), std.detach().cpu().double()
    raw_weight = weight.detach().cpu().double() / std[None, :]
    raw_bias = bias.detach().cpu().double() - raw_weight @ mean
    return (scale * raw_weight).float(), (scale * raw_bias).float()


def fit_regularized_head(
    train_features: Tensor, train_labels: Tensor,
    val_features: Tensor, val_labels: Tensor, *,
    num_classes: int, l2_strength: float, device: torch.device,
    max_iter: int = 100, calibrate: bool = True,
) -> HeadFit:
    """Optimize a frozen-feature multinomial classifier with explicit L2 loss."""
    if (
        isinstance(l2_strength, bool) or not isinstance(l2_strength, (int, float))
        or not math.isfinite(l2_strength) or l2_strength <= 0
    ):
        raise ValueError("l2_strength must be finite and positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(calibrate, bool):
        raise ValueError("calibrate must be bool.")
    _validate_features(train_features, train_labels, val_features, val_labels, num_classes)

    x = train_features.detach().cpu().double()
    xv = val_features.detach().cpu().double()
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-6)
    z, zv = ((x - mean) / std).float().to(device), ((xv - mean) / std).float().to(device)
    y, yv = train_labels.to(device), val_labels.to(device)
    head = nn.Linear(z.shape[1], num_classes).to(device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    optimizer = torch.optim.LBFGS(
        head.parameters(), lr=1.0, max_iter=max_iter, history_size=10,
        line_search_fn="strong_wolfe", tolerance_grad=1e-5, tolerance_change=1e-7,
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(head(z), y)
        loss = loss + 0.5 * l2_strength * head.weight.square().sum()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Head refit objective is not finite.")
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scores = head(zv)
    scale = calibrate_logit_scale(scores, yv) if calibrate else 1.0
    weight, bias = fold_feature_standardization(head.weight, head.bias, mean, std, scale)
    with torch.no_grad():
        train_logits = nn.functional.linear(train_features.float().to(device), weight.to(device), bias.to(device))
        val_logits = nn.functional.linear(val_features.float().to(device), weight.to(device), bias.to(device))
    return HeadFit(
        weight, bias, scale,
        classification_metrics(train_logits, train_labels),
        classification_metrics(val_logits, val_labels),
    )


def fit_discriminant_head(
    train_features: Tensor, train_labels: Tensor,
    val_features: Tensor, val_labels: Tensor, *,
    num_classes: int, shrinkage: float, device: torch.device,
    calibrate: bool = True,
) -> HeadFit:
    """Fit equal-prior LDA with train-only, isotropically shrunk covariance.

    Uses pooled within-class scatter divided by N-K, then
    S_shrunk = (1-a) * S + a * trace(S)/D * I. Shrinkage must be positive.
    Global centering is folded into the bias. A positive validation-fitted
    scale is optional; it does not change the mathematical class ordering.
    """
    _validate_shrinkage(shrinkage)
    if not isinstance(calibrate, bool):
        raise ValueError("calibrate must be bool.")
    _validate_features(train_features, train_labels, val_features, val_labels, num_classes)
    if len(train_labels) <= num_classes:
        raise ValueError("Discriminant fitting needs more training images than classes.")
    x = train_features.detach().cpu().double()
    labels = train_labels.detach().cpu()
    mean = x.mean(0)
    centered = x - mean
    centers = torch.zeros(num_classes, x.shape[1], dtype=torch.float64)
    centers.index_add_(0, labels, centered)
    centers /= torch.bincount(labels, minlength=num_classes)[:, None]
    residuals = centered - centers[labels]
    covariance = residuals.T @ residuals / (len(labels) - num_classes)
    average_variance = covariance.diag().mean()
    if not torch.isfinite(average_variance).item() or average_variance.item() <= 0:
        raise ValueError("Training features need finite, nonzero within-class variation.")
    covariance *= 1 - shrinkage
    covariance.diagonal().add_(shrinkage * average_variance)
    weight = torch.linalg.solve(covariance, centers.T).T
    bias = -0.5 * (centers * weight).sum(1) - weight @ mean
    val_scores = val_features.detach().cpu().double() @ weight.T + bias
    scale = calibrate_logit_scale(val_scores, val_labels) if calibrate else 1.0
    weight, bias = (weight * scale).float(), (bias * scale).float()
    if not torch.isfinite(weight).all().item() or not torch.isfinite(bias).all().item():
        raise FloatingPointError("Discriminant head produced nonfinite coefficients.")
    with torch.no_grad():
        w, b = weight.to(device), bias.to(device)
        train_logits = nn.functional.linear(train_features.float().to(device), w, b)
        val_logits = nn.functional.linear(val_features.float().to(device), w, b)
    return HeadFit(weight, bias, scale,
                   classification_metrics(train_logits, train_labels),
                   classification_metrics(val_logits, val_labels))


def fit_flip_augmented_head(
    train_features: Tensor, train_labels: Tensor,
    val_features: Tensor, val_labels: Tensor, flipped_features: Tensor, *,
    num_classes: int, shrinkage: float, calibrate: bool = True,
) -> HeadFit:
    """Reproduce the selected two-view discriminant fitting recipe on CPU.

    Flipped features must correspond to the same training rows in the same
    order. Centering uses both TRAIN views and is folded into the final layer.
    Float32 centered features match the recorded 320px search's fitting path.
    Reported train metrics cover original images only, not both fitted views.
    """
    _validate_features(train_features, train_labels, val_features, val_labels, num_classes)
    if (
        flipped_features.shape != train_features.shape
        or not flipped_features.is_floating_point()
        or not torch.isfinite(flipped_features).all().item()
    ):
        raise ValueError("flipped_features must match the finite original training features.")
    train_x, train_y = train_features.detach().cpu().float(), train_labels.detach().cpu()
    val_x, val_y = val_features.detach().cpu().float(), val_labels.detach().cpu()
    fit_x = torch.cat((train_x, flipped_features.detach().cpu().float()))
    fit_y = torch.cat((train_y, train_y))
    mean = fit_x.double().mean(0)
    fit = fit_discriminant_head(
        (fit_x.double() - mean).float(), fit_y,
        (val_x.double() - mean).float(), val_y,
        num_classes=num_classes, shrinkage=shrinkage, device=torch.device("cpu"), calibrate=False,
    )
    weight, bias = fold_feature_standardization(
        fit.weight, fit.bias, mean, torch.ones_like(mean),
    )
    scale = calibrate_logit_scale(nn.functional.linear(val_x, weight, bias), val_y) if calibrate else 1.0
    weight, bias = (weight * scale).float(), (bias * scale).float()
    return HeadFit(
        weight, bias, scale,
        classification_metrics(nn.functional.linear(train_x, weight, bias), train_y),
        classification_metrics(nn.functional.linear(val_x, weight, bias), val_y),
    )


@torch.no_grad()
def collect_features(
    model: nn.Module, loader: DataLoader, device: torch.device, *, horizontal_flip: bool = False,
) -> tuple[Tensor, Tensor]:
    """Collect deterministic pooled features without updating the encoder."""
    model.eval()
    features, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        if horizontal_flip:
            images = torch.flip(images, dims=[-1])
        features.append(model.encode(images).cpu())
        labels.append(batch["label"].cpu())
    if not labels:
        raise ValueError("Feature loader produced no samples.")
    return torch.cat(features), torch.cat(labels)


def refit_checkpoint(
    source_path: Path, l2_strength: float | None, device: torch.device, *,
    method: str = "logistic", shrinkage: float | None = None,
    image_size: int | None = None, batch_size: int | None = None,
    train_horizontal_flip: bool = False,
) -> Path:
    """Create an inference-only checkpoint and a local MLflow refit run."""
    from mlflow.tracking import MlflowClient
    from src.data.classification import CubClassificationDataset
    from src.data.preparation import load_classification_manifest
    from src.data.resnet50_transforms import resnet50_transform
    from src.data.source import load_cub_split
    from src.models.resnet50 import ResNet50Classifier
    from src.training.resnet50_experiment import project_path

    if method not in ("logistic", "discriminant"):
        raise ValueError("method must be logistic or discriminant.")
    for name, value in (("image_size", image_size), ("batch_size", batch_size)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be a positive integer when supplied.")
    if not isinstance(train_horizontal_flip, bool) or (train_horizontal_flip and method != "discriminant"):
        raise ValueError("train_horizontal_flip requires discriminant fitting.")
    if method == "discriminant":
        _validate_shrinkage(shrinkage)
        if l2_strength is not None:
            raise ValueError("Discriminant fitting uses shrinkage, not l2_strength.")
    elif (
        shrinkage is not None or isinstance(l2_strength, bool)
        or not isinstance(l2_strength, (int, float))
        or not math.isfinite(l2_strength) or l2_strength <= 0
    ):
        raise ValueError("Logistic fitting requires positive l2_strength and no shrinkage.")
    with source_path.open("rb") as stream:
        source_sha256 = file_digest(stream, "sha256").hexdigest()
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True, mmap=True)
    config = deepcopy(source_checkpoint["config"])
    if image_size is not None:
        config["data"]["image_size"] = image_size
    if batch_size is not None:
        config["data"]["batch_size"] = batch_size
    if config["data"].get("normalization") != "imagenet":
        raise ValueError("This refit requires saved ImageNet normalization.")
    model = ResNet50Classifier(
        num_classes=source_checkpoint["num_classes"], weights=None,
        dropout=config["model"].get("dropout", 0.0),
        finetune_scope=config["model"].get("finetune_scope", "layer4"),
    ).to(device)
    model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
    model.set_stage("head")
    model.eval()
    prepared_dir = project_path(config["data"]["prepared_dir"])
    rows = load_classification_manifest(prepared_dir)
    preparation = json.loads((prepared_dir / "preparation.json").read_text(encoding="utf-8"))
    for key in ("manifest_sha256", "class_names", "source_revisions"):
        if preparation[key] != config["data_provenance"][key]:
            raise ValueError(f"Checkpoint data provenance differs: {key}.")
    source = load_cub_split("train")
    if list(source.class_names) != config["data_provenance"]["class_names"]:
        raise ValueError("Checkpoint class mapping differs from the prepared source.")
    transform = resnet50_transform(
        training=False, image_size=config["data"]["image_size"], geometry=config["data"]["geometry"],
    )
    loaders = {
        part: DataLoader(
            CubClassificationDataset(source, rows, partition=part, transform=transform),
            batch_size=config["data"]["batch_size"], shuffle=False, num_workers=0,
            pin_memory=device.type == "cuda",
        ) for part in ("train", "val")
    }
    tracking = project_path(config["tracking"]["local_dir"])
    tracking.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    name = "resnet50_head_refit"
    experiment = client.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        name, artifact_location=(tracking / "artifacts").as_uri(),
    )
    run = client.create_run(experiment_id, tags={"method": f"frozen_encoder_{method}_head"})
    run_id = run.info.run_id
    folder = project_path(config["checkpoints"]["local_root"]) / "head_refit" / run_id
    try:
        folder.mkdir(parents=True, exist_ok=False)
        print(f"MLflow run: {run_id}\nOutput: {folder}", flush=True)
        features = {}
        for part, loader in loaders.items():
            features[part] = collect_features(model, loader, device)
            print(f"Collected {part} features: {features[part][0].shape}", flush=True)
        if method == "logistic":
            fit = fit_regularized_head(
                *features["train"], *features["val"], num_classes=model.classifier.out_features,
                l2_strength=l2_strength, device=device,
            )
            settings = {
                "method": "l2_logistic_head", "l2_strength": l2_strength,
                "optimizer": "LBFGS", "max_iter": 100,
                "feature_statistics_fit_on": "train", "statistics_folded_into_linear": True,
            }
        else:
            if train_horizontal_flip:
                flipped, flipped_labels = collect_features(model, loaders["train"], device, horizontal_flip=True)
                if not torch.equal(flipped_labels, features["train"][1]):
                    raise ValueError("Flipped training rows must preserve original label order.")
                fit = fit_flip_augmented_head(
                    *features["train"], *features["val"], flipped,
                    num_classes=model.classifier.out_features, shrinkage=shrinkage,
                )
            else:
                fit = fit_discriminant_head(
                    *features["train"], *features["val"], num_classes=model.classifier.out_features,
                    shrinkage=shrinkage, device=device,
                )
            settings = {
                "method": "shrinkage_linear_discriminant", "shrinkage": shrinkage,
                "optimizer": "closed_form_solve", "class_priors": "equal",
                "covariance_divisor": "N-K", "covariance_target": "trace(S)/D * I",
                "feature_statistics_fit_on": "train", "statistics_folded_into_linear": True,
            }
        with torch.no_grad():
            model.classifier.weight.copy_(fit.weight.to(device))
            model.classifier.bias.copy_(fit.bias.to(device))
        metrics = {"train": fit.train_metrics, "val": fit.val_metrics}
        fit_metadata = {
            "run_id": run_id,
            **settings,
            "logit_scale": fit.logit_scale, "calibration_fit_on": "val",
            "source_checkpoint": str(source_path.resolve()),
            "source_sha256": source_sha256,
            "encoder_training_image_size": source_checkpoint["config"]["data"]["image_size"],
            "feature_image_size": config["data"]["image_size"],
            "feature_batch_size": config["data"]["batch_size"],
            "train_views": "original_plus_horizontal_flip" if train_horizontal_flip else "original",
            "training_images": len(features["train"][1]),
            "fit_feature_vectors": len(features["train"][1]) * (2 if train_horizontal_flip else 1),
            "clean_train_metrics": True, "inference_augmentation": False,
            "source_run_id": (
                source_checkpoint.get("head_refit", {}).get("run_id")
                or source_checkpoint.get("config", {}).get("execution", {}).get("run_id")
            ),
        }
        config["head_refit"] = fit_metadata
        payload = {
            "schema_version": 2, "checkpoint_type": "inference_only",
            "training_state_available": False, "num_classes": model.classifier.out_features,
            "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "config": config, "head_refit": fit_metadata, "metrics": metrics,
            "source_stage": source_checkpoint.get("stage"),
            "source_epoch": source_checkpoint.get("epoch"),
        }
        destination = folder / "model.pt"
        with source_path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != source_sha256:
                raise ValueError("Source checkpoint changed during classifier refit.")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".tmp", delete=False) as temporary:
                temporary_path = Path(temporary.name)
                torch.save(payload, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
            temporary_path.replace(destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for key, value in fit_metadata.items():
            client.log_param(run_id, key, value)
        for part, values in metrics.items():
            for key, value in values.items():
                client.log_metric(run_id, f"{part}.{key}", value)
        client.log_artifact(run_id, str(destination), artifact_path="checkpoints")
        client.log_artifact(run_id, str(folder / "metrics.json"), artifact_path="metadata")
        client.log_artifact(run_id, str(folder / "config.json"), artifact_path="metadata")
    except BaseException as error:
        client.set_terminated(run_id, status="KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED")
        raise
    else:
        client.set_terminated(run_id, status="FINISHED")
    print(json.dumps(metrics, indent=2), flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--method", choices=("logistic", "discriminant"), default="logistic")
    parser.add_argument("--l2-strength", type=float)
    parser.add_argument("--shrinkage", type=float)
    parser.add_argument("--image-size", type=int, help="Feature size override; does not retrain the encoder.")
    parser.add_argument("--batch-size", type=int, help="Feature extraction batch size override.")
    parser.add_argument("--train-horizontal-flip", action="store_true", help="Discriminant: fit original + flipped TRAIN views.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.method == "logistic" and (args.l2_strength is None or args.shrinkage is not None):
        parser.error("Logistic fitting requires --l2-strength and no --shrinkage.")
    if args.method == "discriminant" and (args.shrinkage is None or args.l2_strength is not None):
        parser.error("Discriminant fitting requires --shrinkage and no --l2-strength.")
    if args.train_horizontal_flip and args.method != "discriminant":
        parser.error("--train-horizontal-flip requires --method discriminant.")
    for name, value in (("--image-size", args.image_size), ("--batch-size", args.batch_size)):
        if value is not None and value < 1:
            parser.error(f"{name} must be a positive integer.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    refit_checkpoint(
        args.source_checkpoint, args.l2_strength, torch.device(args.device),
        method=args.method, shrinkage=args.shrinkage,
        image_size=args.image_size, batch_size=args.batch_size,
        train_horizontal_flip=args.train_horizontal_flip,
    )


if __name__ == "__main__":
    main()
