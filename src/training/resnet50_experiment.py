"""Two-stage ResNet-50 experiments with local tracking and checkpoints."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.data.classification import make_classification_loaders
from src.models.resnet50 import ResNet50Classifier
from src.training.resnet50_training import (
    build_optimizer,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)


def project_path(value: str) -> Path:
    """Resolve configured relative paths from the repository root."""
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
    """Train both stages and return the directory containing their checkpoints.

    Model and loaders must come from prepare_training with validated config.
    Smoke tests use bounded subsets and a separately named MLflow experiment.
    This starts a new experiment; checkpoint resumption is not implemented here.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise RuntimeError(
            "Install training dependencies with: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-resnet50.txt"
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
