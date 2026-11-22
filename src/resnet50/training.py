"""Two-stage ResNet-50 training loop, checkpoints, and MLflow experiment runner."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Subset

from src.data.classification import make_classification_loaders
from src.resnet50.model import ResNet50Classifier


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def build_optimizer(
    model: ResNet50Classifier,
    *,
    classifier_learning_rate: float,
    weight_decay: float,
    encoder_learning_rate: float | None = None,
) -> AdamW:
    """Build AdamW for the model's active stage (head-only or head + unfrozen encoder)."""
    if (
        not math.isfinite(classifier_learning_rate)
        or classifier_learning_rate <= 0
    ):
        raise ValueError("classifier_learning_rate must be finite and positive.")

    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")

    parameter_groups = [
        {
            "params": model.classifier.parameters(),
            "lr": classifier_learning_rate,
        }
    ]

    if model.stage == "head":
        if encoder_learning_rate is not None:
            raise ValueError("Head training does not use encoder_learning_rate.")
    elif model.stage == "finetune":
        if (
            encoder_learning_rate is None
            or not math.isfinite(encoder_learning_rate)
            or encoder_learning_rate <= 0
        ):
            raise ValueError("encoder_learning_rate must be finite and positive.")

        encoder_parameters = [
            parameter for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ]
        if not encoder_parameters:
            raise ValueError("Fine-tuning requires trainable encoder parameters.")
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_learning_rate,
            }
        )
    else:
        raise ValueError(f"Unsupported training stage: {model.stage!r}")

    return AdamW(
        parameter_groups,
        lr=classifier_learning_rate,
        weight_decay=weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    *,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
) -> EpochMetrics:
    """Run one training epoch with optional label smoothing and MixUp."""
    if (
        isinstance(label_smoothing, bool)
        or not isinstance(label_smoothing, (int, float))
        or not 0 <= label_smoothing < 1
    ):
        raise ValueError("label_smoothing must be a finite probability in [0, 1).")
    if (
        isinstance(mixup_alpha, bool) or not isinstance(mixup_alpha, (int, float))
        or not math.isfinite(mixup_alpha) or mixup_alpha < 0
    ):
        raise ValueError("mixup_alpha must be finite and non-negative.")
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    loss_sum = 0.0
    correct = 0
    samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        weight = 1.0
        mixed_labels = labels
        if mixup_alpha > 0:
            weight = float(np.random.beta(mixup_alpha, mixup_alpha))
            order = torch.randperm(images.size(0), device=images.device)
            images = weight * images + (1 - weight) * images[order]
            mixed_labels = labels[order]
        logits = model(images)
        loss = loss_fn(logits, labels)
        if mixup_alpha > 0:
            loss = weight * loss + (1 - weight) * loss_fn(logits, mixed_labels)

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Training loss is not finite.")

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += weight * (predictions == labels).sum().item()
        correct += (1 - weight) * (predictions == mixed_labels).sum().item()
        samples += batch_size

    if samples == 0:
        raise ValueError("The training loader produced no samples.")

    return EpochMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
    )


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochMetrics:
    """Measure validation loss and accuracy."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        if not torch.isfinite(loss).item():
            raise FloatingPointError("Validation loss is not finite.")

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += batch_size

    if samples == 0:
        raise ValueError("The validation loader produced no samples.")

    return EpochMetrics(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
    )


def save_checkpoint(
    path: str | Path,
    *,
    model: ResNet50Classifier,
    optimizer: Optimizer,
    epoch: int,
    val_metrics: EpochMetrics,
    best_val_accuracy: float,
    config: dict[str, object],
) -> None:
    """Atomically save a stage checkpoint with model, optimizer, metrics, and config."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer.")

    if (
        val_metrics.samples < 1
        or not math.isfinite(val_metrics.loss)
        or val_metrics.loss < 0
        or not 0 <= val_metrics.accuracy <= 1
    ):
        raise ValueError("Validation metrics must be finite and valid.")

    if not val_metrics.accuracy <= best_val_accuracy <= 1:
        raise ValueError(
            "best_val_accuracy must be between current validation accuracy and 1."
        )

    checkpoint = {
        "schema_version": 1,
        "num_classes": model.classifier.out_features,
        "stage": model.stage,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": asdict(val_metrics),
        "best_val_accuracy": best_val_accuracy,
        "config": json.loads(json.dumps(config, allow_nan=False)),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            torch.save(checkpoint, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def project_path(value: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


def _flatten_parameters(settings: dict[str, Any], prefix: str = "") -> dict[str, str]:
    parameters = {}
    for key, value in settings.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            parameters.update(_flatten_parameters(value, name))
        else:
            parameters[name] = str(value)
    return parameters


def run_experiment(
    model: ResNet50Classifier,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
    *,
    smoke_test: bool = False,
) -> Path:
    """Run head warmup followed by encoder fine-tuning and log to local MLflow."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError(
            "Install training dependencies with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from error

    config = deepcopy(config)
    mode = "smoke" if smoke_test else "training"
    if smoke_test:
        config["experiment"]["name"] += "_smoke"
        config["data"]["batch_size"] = min(config["data"]["batch_size"], 4)
        for stage in ("head", "finetune"):
            config["training"][stage]["epochs"] = 1
        sample_limit = 2 * config["data"]["batch_size"]
        datasets = {
            name: Subset(loader.dataset, range(min(len(loader.dataset), sample_limit)))
            for name, loader in loaders.items()
        }
        loaders = make_classification_loaders(
            datasets,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            seed=config["experiment"]["seed"],
            pin_memory=device.type == "cuda",
        )

    preparation_path = project_path(config["data"]["prepared_dir"]) / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    parameters = _flatten_parameters(config)
    parameters.update({
        "execution.mode": mode,
        "data.manifest_sha256": preparation["manifest_sha256"],
        "data.train_samples": str(len(loaders["train"].dataset)),
        "data.val_samples": str(len(loaders["val"].dataset)),
        "environment.torch": torch.__version__,
    })
    config["data_provenance"] = {
        key: preparation[key]
        for key in ("manifest_sha256", "class_names", "source_revisions")
    }

    tracking_dir = project_path(config["tracking"]["local_dir"])
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = config["experiment"]["name"]
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment is not None
        else client.create_experiment(
            experiment_name,
            artifact_location=(tracking_dir / "artifacts").as_uri(),
        )
    )
    run = client.create_run(experiment_id, tags={
        "mlflow.runName": f"{mode}_seed_{config['experiment']['seed']}",
        "execution.mode": mode,
    })
    run_id = run.info.run_id
    run_dir = project_path(config["checkpoints"]["local_root"]) / mode / run_id

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        config["execution"] = {"mode": mode, "run_id": run_id}
        config_path = run_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        for name, value in parameters.items():
            client.log_param(run_id, name, value)
        client.log_param(run_id, "checkpoints.directory", str(run_dir))
        client.log_artifact(run_id, str(config_path), artifact_path="metadata")
        print(f"MLflow run: {run_id}\nCheckpoints: {run_dir}", flush=True)

        history = []
        best_accuracy = -1.0
        best_record = None
        global_epoch = 0
        stopping_record = None
        for stage_index, stage in enumerate(("head", "finetune")):
            model.set_stage(stage)
            settings = config["training"][stage]
            accuracy_label = (
                "mixup target agreement" if settings.get("mixup_alpha", 0.0) > 0
                else "train accuracy"
            )
            optimizer_arguments = {
                "classifier_learning_rate": (
                    settings["learning_rate"] if stage == "head"
                    else settings["classifier_learning_rate"]
                ),
                "weight_decay": config["training"]["weight_decay"],
            }
            if stage == "finetune":
                optimizer_arguments["encoder_learning_rate"] = settings["encoder_learning_rate"]
            optimizer = build_optimizer(model, **optimizer_arguments)
            client.set_tag(run_id, "current_stage", stage)
            client.log_param(
                run_id, f"{stage}.trainable_parameters",
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            )
            client.set_tag(
                run_id, f"{stage}.accuracy_definition",
                "mixup_target_agreement" if settings.get("mixup_alpha", 0.0) > 0
                else "hard_label_accuracy",
            )
            stage_best = -1.0
            stopping = settings.get("early_stopping") if stage == "finetune" else None
            best_stopping_loss = float("inf")
            epochs_without_improvement = 0

            for epoch in range(1, settings["epochs"] + 1):
                global_epoch += 1
                print(f"Starting {stage} epoch {epoch}/{settings['epochs']}...", flush=True)
                started = perf_counter()
                train = train_one_epoch(
                    model, loaders["train"], optimizer, device,
                    label_smoothing=settings.get("label_smoothing", config["training"].get("label_smoothing", 0.0)),
                    mixup_alpha=settings.get("mixup_alpha", 0.0),
                )
                val = validate_one_epoch(model, loaders["val"], device)
                seconds = perf_counter() - started
                metrics = {
                    "train_loss": train.loss, "train_accuracy": train.accuracy,
                    "val_loss": val.loss, "val_accuracy": val.accuracy,
                    "epoch_seconds": seconds, "stage_epoch": epoch,
                    "stage_index": stage_index, f"{stage}.val_accuracy": val.accuracy,
                }
                for name, value in metrics.items():
                    client.log_metric(run_id, name, value, step=global_epoch)
                record = {
                    "stage": stage, "epoch": epoch, "global_epoch": global_epoch,
                    "train": asdict(train), "val": asdict(val), "seconds": seconds,
                }
                history.append(record)
                improved = val.accuracy > best_accuracy
                if improved:
                    best_accuracy, best_record = val.accuracy, record
                checkpoint_arguments = dict(
                    model=model, optimizer=optimizer, epoch=epoch,
                    val_metrics=val, best_val_accuracy=best_accuracy, config=config,
                )
                save_checkpoint(run_dir / "last.pt", **checkpoint_arguments)
                if improved:
                    save_checkpoint(run_dir / "best.pt", **checkpoint_arguments)
                if val.accuracy > stage_best:
                    stage_best = val.accuracy
                    save_checkpoint(run_dir / f"{stage}_best.pt", **checkpoint_arguments)
                print(
                    f"{stage} epoch {epoch}: train loss={train.loss:.4f}, "
                    f"{accuracy_label}={train.accuracy:.2%}, "
                    f"val loss={val.loss:.4f}, val accuracy={val.accuracy:.2%}",
                    flush=True,
                )
                if stopping:
                    if val.loss < best_stopping_loss - stopping.get("min_delta", 0.0):
                        best_stopping_loss = val.loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    client.log_metric(
                        run_id, "finetune.epochs_without_improvement",
                        epochs_without_improvement, step=global_epoch,
                    )
                    if epochs_without_improvement >= stopping["patience"]:
                        stopping_record = {
                            "stage": stage, "epoch": epoch, "monitor": "val_loss",
                            "best_loss": best_stopping_loss,
                            "reason": "patience_exhausted",
                        }
                        client.set_tag(run_id, "early_stopping.reason", "patience_exhausted")
                        print(f"Early stopping at {stage} epoch {epoch}.", flush=True)
                        break

        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps({
            "run_id": run_id, "mode": mode, "best": best_record, "history": history,
            "early_stopping": stopping_record,
        }, indent=2), encoding="utf-8")
        client.log_metric(run_id, "best_val_accuracy", best_accuracy, step=global_epoch)
        client.log_artifact(run_id, str(summary_path), artifact_path="metadata")
        client.log_artifact(run_id, str(run_dir / "best.pt"), artifact_path="checkpoints")
    except BaseException as error:
        status = "KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        client.set_terminated(run_id, status=status)
        raise
    else:
        client.set_terminated(run_id, status="FINISHED")

    print(f"{mode.capitalize()} run complete. Best validation accuracy: {best_accuracy:.2%}")
    return run_dir
