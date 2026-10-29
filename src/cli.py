"""Unified Command Line Interface for Deep Learning Final Project."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch
import yaml

from src.data.classification import CubClassificationDataset, make_classification_loaders
from src.data.image_io import load_rgb
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.transforms import classification_transform
from src.evaluation.metrics import confusion_matrix, summarize_classification
from src.models.scratch import ScratchConfig, create_scratch_model
from src.training.trainer import (
    TrainingConfig,
    evaluate_classification,
    train_classification,
)
from src.training.tuning import compare_tuning, grid_search, random_search
from src.visualization.gradcam import collect_gradcam_samples, plot_gradcam_grid
from src.visualization.training_plots import (
    plot_confusion_summary,
    plot_training_curves,
    plot_tuning_comparison,
)


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_loaders_from_config(config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    data_cfg = config.get("data", {})
    prepared_dir = Path(data_cfg.get("prepared_dir", "outputs/classification-v1"))
    manifest = load_classification_manifest(prepared_dir)
    source = load_cub_split("train")
    strong_augment = bool(data_cfg.get("strong_augment", True))

    datasets = {
        "train": CubClassificationDataset(
            source,
            manifest,
            partition="train",
            transform=classification_transform(training=True, strong_augment=strong_augment),
        ),
        "val": CubClassificationDataset(
            source,
            manifest,
            partition="val",
            transform=classification_transform(training=False),
        ),
    }

    loaders = make_classification_loaders(
        datasets,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        seed=int(data_cfg.get("seed", 42)),
        pin_memory=torch.cuda.is_available(),
    )
    return loaders, source


def _cmd_train(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    # If run_dir, prepared_dir or JSON config is supplied, route to pretrained experiment runner
    if args.run_dir is not None or args.prepared_dir is not None or config_path.suffix == ".json":
        from src.training.runtime import _failure
        try:
            from src.training.experiment import run_training
            config = json.loads(config_path.read_text(encoding="utf-8"))
            device = args.device or "cuda"
            prepared_dir = args.prepared_dir or Path("outputs/preparation")
            run_dir = args.run_dir or Path("outputs/classification_run")
            summary = run_training(
                prepared_dir,
                run_dir,
                config,
                device=device,
                resume=args.resume,
                tracking=args.tracking,
            )
            print(json.dumps(summary, allow_nan=False))
            return 0
        except KeyboardInterrupt:
            print("Training interrupted; resume from the latest completed checkpoint.", file=sys.stderr)
            return 130
        except Exception as error:
            failure = _failure("cli", error)
            print(f"Training error: {failure['type']}: {failure['message']}", file=sys.stderr)
            return 1

    # Otherwise execute scratch baseline training workflow
    print(f"Loading configuration from {args.config}...")
    cfg = _load_yaml_config(args.config)
    output_dir = Path(cfg.get("output_dir", "outputs/classification_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders, source = _build_loaders_from_config(cfg)
    model_cfg_data = cfg.get("model", {})
    model_config = ScratchConfig(
        num_classes=int(model_cfg_data.get("num_classes", 200)),
        head_dropout=float(model_cfg_data.get("head_dropout", 0.1)),
        activation=str(model_cfg_data.get("activation", "relu")),
    )
    print(f"Instantiating from-scratch ResNet ({model_config.activation} activation)...")
    model = create_scratch_model(model_config)

    train_cfg_data = cfg.get("training", {})
    train_config = TrainingConfig(
        epochs=int(train_cfg_data.get("epochs", 20)),
        learning_rate=float(train_cfg_data.get("learning_rate", 3e-4)),
        weight_decay=float(train_cfg_data.get("weight_decay", 1e-4)),
        label_smoothing=float(train_cfg_data.get("label_smoothing", 0.1)),
        seed=int(train_cfg_data.get("seed", 42)),
        optimizer=str(train_cfg_data.get("optimizer", "adam")),
        amp=bool(train_cfg_data.get("amp", True)),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Starting training on {device} using {train_config.optimizer} for {train_config.epochs} epochs...")

    def on_epoch(report):
        print(
            f"Epoch {report.epoch:02d}/{train_config.epochs:02d} | "
            f"Train Loss: {report.train_loss:.4f} Acc: {report.train_accuracy:.2%} | "
            f"Val Loss: {report.val_loss:.4f} Acc: {report.val_accuracy:.2%}"
        )

    result = train_classification(model, loaders, train_config, device=device, on_epoch=on_epoch)

    print(f"\nTraining completed! Best Epoch: {result.best_epoch} with Val Acc: {result.best_val_accuracy:.2%}")

    # Save checkpoint
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": result.best_state,
            "model_config": model_config,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
        },
        checkpoint_path,
    )
    print(f"Saved best model weights to {checkpoint_path}")

    # Save training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(result.reports, curves_path)
    print(f"Saved training curves to {curves_path}")

    # Final evaluation on validation set with best weights
    model.load_state_dict(result.best_state)
    val_loss, val_matrix = evaluate_classification(model, loaders["val"], device=device)
    metrics = summarize_classification(val_matrix)
    print("\n--- Final Validation Metrics ---")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {metrics['macro_f1']:.4f}")

    # Save metrics JSON and confusion matrix
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_summary(val_matrix, output_dir / "confusion_matrix.png", class_names=source.class_names)
    print(f"Saved confusion matrix analysis to {output_dir / 'confusion_matrix.png'}")
    return 0


def _cmd_tune(args: argparse.Namespace) -> None:
    print(f"Loading tuning configuration from {args.config}...")
    cfg = _load_yaml_config(args.config)
    output_dir = Path(cfg.get("output_dir", "outputs/classification_run")) / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders, _ = _build_loaders_from_config(cfg)
    tuning_cfg = cfg.get("tuning", {})
    epochs_per_trial = int(tuning_cfg.get("epochs_per_trial", 2))
    random_trials = int(tuning_cfg.get("random_trials", 4))
    param_grid = tuning_cfg.get(
        "grid",
        {
            "optimizer": ["adam", "sgd", "adagrad"],
            "learning_rate": [1e-3, 3e-4],
            "activation": ["relu", "leaky_relu"],
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Running Grid Search ({epochs_per_trial} epochs/trial) ---")
    grid_res = grid_search(param_grid, loaders, epochs=epochs_per_trial, device=device)
    print(f"Grid Search Done! Best Val Acc: {grid_res.best_trial.best_val_accuracy:.2%} in {grid_res.total_time_seconds:.1f}s")

    print(f"\n--- Running Random Search ({random_trials} trials, {epochs_per_trial} epochs/trial) ---")
    rand_res = random_search(param_grid, loaders, num_trials=random_trials, epochs=epochs_per_trial, device=device)
    print(f"Random Search Done! Best Val Acc: {rand_res.best_trial.best_val_accuracy:.2%} in {rand_res.total_time_seconds:.1f}s")

    comparison = compare_tuning(grid_res, rand_res)
    with open(output_dir / "tuning_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    plot_tuning_comparison(comparison, output_dir / "tuning_comparison.png")
    print(f"\nSaved tuning comparison plot to {output_dir / 'tuning_comparison.png'}")
    print(json.dumps(comparison, indent=2))


def _cmd_gradcam(args: argparse.Namespace) -> None:
    cfg = _load_yaml_config(args.config)
    output_dir = Path(cfg.get("output_dir", "outputs/classification_run"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train first or specify --checkpoint.")

    print(f"Loading weights from {checkpoint_path} for Grad-CAM...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", ScratchConfig())
    model = create_scratch_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    loaders, source = _build_loaders_from_config(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Collecting correct and wrong predictions with Grad-CAM overlays...")
    samples = collect_gradcam_samples(
        model,
        loaders["val"],
        device=device,
        class_names=source.class_names,
        max_correct=3,
        max_wrong=3,
    )

    out_path = output_dir / "gradcam_attention.png"
    plot_gradcam_grid(samples, out_path)
    print(f"Saved Grad-CAM attention comparison (correct vs wrong) to {out_path}")


def _cmd_predict(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("outputs/classification_run/best_model.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", ScratchConfig())
    model = create_scratch_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    source = load_cub_split("train")
    transform = classification_transform(training=False)

    with Image.open(args.image) as raw_img:
        tensor = transform(raw_img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_id = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_id].item())

    print(f"Prediction: {source.class_names[pred_id]} (Class ID: {pred_id})")
    print(f"Confidence: {confidence:.2%}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FeatherLens Multimodal Deep Learning CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. train
    p_train = subparsers.add_parser("train", help="Train a model")
    p_train.add_argument("--task", default="classification", choices=["classification"])
    p_train.add_argument("--config", default="configs/cls.yaml", help="Path to YAML or JSON config")
    p_train.add_argument("--prepared-dir", type=Path, default=None, help="Path to prepared split directory")
    p_train.add_argument("--run-dir", type=Path, default=None, help="Path to run output directory")
    p_train.add_argument("--device", default=None, help="Computation device (e.g., cpu, cuda)")
    p_train.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    p_train.add_argument("--tracking", choices=["mlflow", "json"], default="mlflow", help="Experiment tracking")

    # 2. tune
    p_tune = subparsers.add_parser("tune", help="Run hyperparameter tuning (Grid & Random Search)")
    p_tune.add_argument("--task", default="classification", choices=["classification"])
    p_tune.add_argument("--config", default="configs/cls.yaml", help="Path to YAML config")

    # 3. gradcam
    p_cam = subparsers.add_parser("gradcam", help="Generate Grad-CAM explainability heatmaps")
    p_cam.add_argument("--task", default="classification", choices=["classification"])
    p_cam.add_argument("--config", default="configs/cls.yaml", help="Path to YAML config")
    p_cam.add_argument("--checkpoint", default=None, help="Path to trained checkpoint")

    # 4. predict
    p_pred = subparsers.add_parser("predict", help="Predict class for a single image")
    p_pred.add_argument("--task", default="classification")
    p_pred.add_argument("--image", required=True, help="Path to input image")
    p_pred.add_argument("--checkpoint", default=None, help="Path to model checkpoint")

    args = parser.parse_args(argv)
    if args.command == "train":
        return _cmd_train(args)
    elif args.command == "tune":
        _cmd_tune(args)
        return 0
    elif args.command == "gradcam":
        _cmd_gradcam(args)
        return 0
    elif args.command == "predict":
        _cmd_predict(args)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
