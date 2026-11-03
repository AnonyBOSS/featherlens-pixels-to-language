"""Unified Command Line Interface for Deep Learning Final Project."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image
import torch

from src.data.classification import CubClassificationDataset, make_classification_loaders
from src.data.image_io import load_rgb
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.transforms import classification_transform
from src.evaluation.metrics import confusion_matrix, summarize_classification
from src.models.scratch import ScratchConfig, create_scratch_model
from src.training.scratch_trainer import (
    TrainingConfig,
    evaluate_classification,
    train_classification,
)
from src.training.scratch_trainer import _seed_everything
from src.training.tuning import compare_tuning, grid_search, random_search
from src.visualization.gradcam import collect_gradcam_samples, plot_gradcam_grid
from src.visualization.training_plots import (
    plot_confusion_summary,
    plot_training_curves,
    plot_tuning_comparison,
)


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    import yaml
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
    if (getattr(args, "run_dir", None) is not None
            or getattr(args, "prepared_dir", None) is not None or config_path.suffix == ".json"):
        from src.training.runtime import _failure
        try:
            from src.training.experiment import run_training
            config = json.loads(config_path.read_text(encoding="utf-8"))
            device = getattr(args, "device", None) or "cuda"
            prepared_dir = getattr(args, "prepared_dir", None) or Path("outputs/preparation")
            run_dir = getattr(args, "run_dir", None) or Path("outputs/classification_run")
            initial_checkpoint = getattr(args, "initial_checkpoint", None)
            options = ({"initial_checkpoint": initial_checkpoint} if initial_checkpoint is not None else {})
            summary = run_training(
                prepared_dir,
                run_dir,
                config,
                device=device,
                resume=getattr(args, "resume", False),
                tracking=getattr(args, "tracking", "mlflow"),
                **options,
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

    # Scratch checkpoints contain weights only, so optimizer recovery is unavailable.
    if getattr(args, "resume", False) or getattr(args, "initial_checkpoint", None) is not None:
        raise ValueError("Scratch training does not support --resume or --initial-checkpoint")

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

    _seed_everything(train_config.seed)
    print(f"Instantiating from-scratch ResNet ({model_config.activation} activation)...")
    model = create_scratch_model(model_config)

    device = torch.device(getattr(args, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu"))
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


def _cmd_evaluate(args: argparse.Namespace) -> None:
    if args.split == "test" and not getattr(args, "allow_official_test", False):
        raise ValueError("Evaluating official test requires --allow-official-test explicitly")
    print(f"Loading configuration from {args.config}...")
    cfg = _load_yaml_config(args.config)
    output_dir = Path(cfg.get("output_dir", "outputs/classification_run"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", ScratchConfig())
    model = create_scratch_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    split_name = args.split
    data_cfg = cfg.get("data", {})
    prepared_dir = Path(data_cfg.get("prepared_dir", "outputs/classification-v1"))
    manifest = load_classification_manifest(prepared_dir)
    source = load_cub_split("test" if split_name == "test" else "train")

    dataset = CubClassificationDataset(
        source,
        manifest,
        partition=split_name,
        transform=classification_transform(training=False),
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Evaluating {checkpoint_path.name} on '{split_name}' split ({len(dataset)} samples)...")
    loss, matrix = evaluate_classification(model, loader, device=device)
    metrics = summarize_classification(matrix)

    print(f"\n--- {split_name.capitalize()} Evaluation Metrics ---")
    print(f"Loss:            {loss:.4f}")
    print(f"Accuracy:        {metrics['accuracy']:.4%}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {metrics['macro_f1']:.4f}")

    metrics["loss"] = loss
    metrics["num_samples"] = len(dataset)
    metrics_file = output_dir / f"{split_name}_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")


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


def _delivery_runtime():
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Match the measured FP32 Base encoder: cuDNN may use TF32 kernels.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False


def _cmd_bundle(args: argparse.Namespace) -> dict[str, Any]:
    _delivery_runtime()
    from src.evaluation.inference import export_classifier_bundle, load_classifier
    if args.command == 'export-classifier':
        path=export_classifier_bundle(args.tiny_checkpoint,args.base_weights,args.base_head,args.output_dir)
        result={'status':'exported','bundle':str(path)}
    else:
        classifier=load_classifier(args.bundle,device=(args.device or "cpu"))
        from src.evaluation.classification import predict_image,evaluate_classifier
        if args.command == 'predict':
            result=predict_image(classifier,args.image,top_k=args.top_k)
        elif args.command in ('evaluate', 'test'):
            report=evaluate_classifier(classifier,args.prepared_dir,partition=args.split,
                batch_size=args.batch_size,num_workers=args.num_workers,
                allow_test=args.allow_official_test,output_dir=args.output_dir)
            result={k:v for k,v in report.items() if k not in
                ('predictions','per_class','confusion_matrix','class_names')}
        else:
            from src.data.image_io import load_rgb
            from src.evaluation.gradcam import compute_gradcam,render_gradcam
            if args.component not in classifier.models:
                raise ValueError('Unknown component name in classifier bundle')
            if args.output.exists() or args.output.with_suffix('.json').exists():
                raise FileExistsError('Grad-CAM output already exists; preserve previous results')
            prediction=predict_image(classifier,args.image)
            target=prediction['class_id'] if args.target_class is None else args.target_class
            with load_rgb({'path':str(args.image)}) as image:
                recipe=classifier.preprocessing[args.component]
                tensor=recipe.build_transform(training=False)(image).unsqueeze(0).to(classifier.device)
                computed=compute_gradcam(classifier.models[args.component],tensor,target)
                render_gradcam(image,computed['heatmap'],recipe.image_size,output_path=args.output)
            result={k:v for k,v in computed.items() if k not in ('heatmap','logits')}
            result.update(status='written',component=args.component,target_class_name=classifier.class_names[target],
                component_prediction_name=classifier.class_names[computed['predicted_class']],
                classifier_prediction=prediction,output=args.output.name,
                interpretation='Raw component-score attribution; relative colors are not causal proof or ensemble contributions.')
            args.output.with_suffix('.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FeatherLens classification CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train scratch YAML or pretrained JSON classification")
    train.add_argument("--task", default="classification", choices=["classification"])
    train.add_argument("--config", type=Path, required=True, help="Scratch YAML or pretrained JSON config")
    train.add_argument("--prepared-dir", type=Path)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--device")
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument("--resume", action="store_true")
    initialization.add_argument("--initial-checkpoint", type=Path,
                                help="Initialize a fresh pretrained run from model weights only")
    train.add_argument("--tracking", choices=["mlflow", "json"], default="mlflow")

    tune = commands.add_parser("tune", help="Run scratch Grid Search and Random Search")
    tune.add_argument("--task", default="classification", choices=["classification"])
    tune.add_argument("--config", default="configs/cls.yaml")

    export = commands.add_parser("export-classifier", help="Package trained Tiny and scaled Base for offline inference")
    export.add_argument("--tiny-checkpoint", type=Path, required=True)
    export.add_argument("--base-weights", type=Path, required=True)
    export.add_argument("--base-head", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--device", default="cpu")

    predict = commands.add_parser("predict", help="Classify one image from a scratch checkpoint or portable bundle")
    cam = commands.add_parser("gradcam", help="Explain scratch validation samples or a bundle component")
    evaluate_commands = []
    for name in ("evaluate", "test"):
        command = commands.add_parser(name, help="Evaluate a complete partition; official test requires explicit opt-in")
        command.add_argument("--config", default="configs/cls.yaml")
        command.add_argument("--split", "--partition", dest="split", choices=["train", "val", "test"],
                             default="test" if name == "test" else "val")
        command.add_argument("--prepared-dir", type=Path)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--num-workers", type=int, default=0)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--allow-official-test", action="store_true",
                             help="Only for an explicitly frozen final test protocol")
        evaluate_commands.append(command)
    for command in (predict, cam, *evaluate_commands):
        command.add_argument("--task", default="classification", choices=["classification"])
        inputs = command.add_mutually_exclusive_group()
        inputs.add_argument("--checkpoint", type=Path, help="Scratch best_model.pt checkpoint")
        inputs.add_argument("--bundle", type=Path, help="Portable pretrained classifier bundle")
        command.add_argument("--device", help="Bundle default: cpu; scratch evaluation default: available device")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--top-k", type=int, default=5)
    cam.add_argument("--config", default="configs/cls.yaml")
    cam.add_argument("--image", type=Path)
    cam.add_argument("--component", help="Component name in the bundle, e.g. tiny or base")
    cam.add_argument("--target-class", type=int)
    cam.add_argument("--output", type=Path, help="Bundle PNG overlay path; JSON metadata is saved beside it")

    args = parser.parse_args(argv)
    if getattr(args, "bundle", None) is not None:
        required = (("prepared_dir", "output_dir") if args.command in ("evaluate", "test")
                    else ("image", "component", "output") if args.command == "gradcam" else ())
        missing = ["--" + name.replace("_", "-") for name in required if getattr(args, name) is None]
        if missing:
            parser.error("bundle mode requires " + ", ".join(missing))
    try:
        if args.command in ("evaluate", "test") and args.split == "test" and not args.allow_official_test:
            raise ValueError("Evaluating official test requires --allow-official-test explicitly")
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "export-classifier" or getattr(args, "bundle", None) is not None:
            print(json.dumps(_cmd_bundle(args), allow_nan=False))
        elif args.command == "tune":
            _cmd_tune(args)
        elif args.command == "gradcam":
            _cmd_gradcam(args)
        elif args.command == "predict":
            _cmd_predict(args)
        elif args.command in ("evaluate", "test"):
            _cmd_evaluate(args)
        return 0
    except KeyboardInterrupt:
        message = ("Training interrupted; resume from the latest completed checkpoint."
                   if args.command == "train" else "Classification operation interrupted.")
        print(message, file=sys.stderr)
        return 130
    except Exception as error:
        prefix = "Training" if args.command == "train" else "Classification"
        print(f"{prefix} error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
