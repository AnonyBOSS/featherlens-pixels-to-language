"""Evaluate fixed ResNet-50 checkpoints on the full official CUB test split.

No training, checkpoint selection, confidence calibration or test augmentation
is performed. Each evaluation writes an independent report and local MLflow run.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def metrics_from_confusion(matrix: Tensor) -> dict:
    """Rows are true classes; columns are predictions; undefined ratios are zero."""
    matrix = matrix.detach().cpu().double()
    if (
        matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2 or not torch.isfinite(matrix).all()
        or (matrix < 0).any() or matrix.sum() <= 0
    ):
        raise ValueError("Expected a nonempty square confusion matrix of counts.")
    true_count, predicted_count = matrix.sum(1), matrix.sum(0)
    correct = matrix.diag()
    precision = correct / predicted_count.clamp_min(1)
    recall = correct / true_count.clamp_min(1)
    f1 = 2 * correct / (true_count + predicted_count).clamp_min(1)
    return {
        "samples": int(matrix.sum().item()),
        "correct": int(correct.sum().item()),
        "accuracy": (correct.sum() / matrix.sum()).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class": [
            {"class_id": index, "support": int(true_count[index].item()),
             "precision": precision[index].item(), "recall": recall[index].item(),
             "f1": f1[index].item()}
            for index in range(matrix.shape[0])
        ],
    }


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module, loader: DataLoader, *, device: torch.device,
    num_classes: int, expected_samples: int,
) -> tuple[dict, Tensor, list[dict]]:
    """Use all samples exactly once, in eval mode, with no gradient updates."""
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    predictions = []
    seen = set()
    loss_sum = 0.0
    for batch_number, batch in enumerate(loader, 1):
        labels = batch["label"].to(device)
        logits = model(batch["image"].to(device))
        if (
            logits.shape != (labels.numel(), num_classes)
            or not torch.isfinite(logits).all()
            or labels.ndim != 1 or labels.dtype != torch.int64
            or (labels < 0).any() or (labels >= num_classes).any()
        ):
            raise ValueError("Invalid evaluation logits or labels.")
        loss_sum += nn.functional.cross_entropy(logits.double(), labels, reduction="sum").item()
        predicted = logits.argmax(1)
        confidence = logits.softmax(1).max(1).values.cpu().tolist()
        truth, guessed = labels.cpu(), predicted.cpu()
        matrix += torch.bincount(
            truth * num_classes + guessed, minlength=num_classes**2,
        ).reshape(num_classes, num_classes)
        identifiers = batch["image_id"]
        if len(identifiers) != labels.numel():
            raise ValueError("Image identity count differs from batch size.")
        for identifier, actual, guess, probability in zip(identifiers, truth.tolist(), guessed.tolist(), confidence):
            if identifier in seen:
                raise ValueError("Repeated test image; evaluation must cover every image once.")
            seen.add(identifier)
            predictions.append({"image_id": identifier, "true_class": actual,
                                "predicted_class": guess, "confidence": probability})
        if batch_number % 20 == 0:
            print(f"Evaluated {len(predictions)}/{expected_samples} test images...", flush=True)
    if len(predictions) != expected_samples:
        raise ValueError(f"Incomplete evaluation: expected {expected_samples}, got {len(predictions)}.")
    metrics = metrics_from_confusion(matrix)
    metrics["cross_entropy"] = loss_sum / expected_samples
    return metrics, matrix, predictions


def evaluate_checkpoint(checkpoint_path: Path, *, device: torch.device, output_root: Path) -> Path:
    from mlflow.tracking import MlflowClient
    from src.data.classification import CubClassificationDataset
    from src.data.preparation import load_classification_manifest
    from src.data.resnet50_transforms import resnet50_transform
    from src.data.source import load_cub_split
    from src.models.resnet50 import ResNet50Classifier
    from src.training.resnet50_experiment import project_path

    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    config = checkpoint["config"]
    prepared_dir = project_path(config["data"]["prepared_dir"])
    manifest = load_classification_manifest(prepared_dir)
    preparation = json.loads((prepared_dir / "preparation.json").read_text(encoding="utf-8"))
    for key in ("manifest_sha256", "class_names", "source_revisions"):
        if preparation[key] != config["data_provenance"][key]:
            raise ValueError(f"Checkpoint preparation provenance differs: {key}.")
    if config["data"].get("normalization") != "imagenet":
        raise ValueError("This evaluator requires saved ImageNet normalization.")
    source = load_cub_split("test")
    if list(source.class_names) != preparation["class_names"]:
        raise ValueError("Official test class ordering differs from checkpoint ordering.")
    dataset = CubClassificationDataset(source, manifest, partition="test", transform=resnet50_transform(
        training=False, image_size=config["data"]["image_size"], geometry=config["data"]["geometry"],
    ))
    expected = preparation["source_counts"]["test"]
    if len(dataset) != expected or len(source.dataset) != expected:
        raise ValueError("Evaluation must include the full official test population.")
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False,
                        num_workers=0, drop_last=False, pin_memory=device.type == "cuda")
    model = ResNet50Classifier(
        num_classes=checkpoint["num_classes"], weights=None,
        dropout=config["model"].get("dropout", 0),
        finetune_scope=config["model"].get("finetune_scope", "layer4"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    digest = sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    tracking = project_path(config["tracking"]["local_dir"])
    tracking.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    experiment = client.get_experiment_by_name("resnet50_official_test")
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        "resnet50_official_test", artifact_location=(tracking / "artifacts").as_uri(),
    )
    source_run = checkpoint.get("head_refit", {}).get("run_id") or config.get("execution", {}).get("run_id")
    run = client.create_run(experiment_id, tags={"purpose": "fixed_checkpoint_official_test", "source_run_id": str(source_run)})
    run_id = run.info.run_id
    folder = output_root.resolve() / run_id
    try:
        folder.mkdir(parents=True, exist_ok=False)
        print(f"Testing: {checkpoint_path}\nMLflow evaluation run: {run_id}", flush=True)
        metrics, matrix, predictions = evaluate_loader(
            model, loader, device=device, num_classes=checkpoint["num_classes"], expected_samples=expected,
        )
        report = {
            "schema_version": 1, "partition": "official_test", "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": digest.hexdigest(), "source_run_id": source_run,
            "evaluation_run_id": run_id, "device": str(device), "torch_version": str(torch.__version__),
            "class_names": list(source.class_names), "data_provenance": config["data_provenance"],
            "preprocessing": {key: config["data"][key] for key in ("image_size", "geometry", "normalization")},
            "test_time_augmentation": False, "weights_updated": False,
            "calibration_refitted_on_test": False,
            "selection": "Checkpoint fixed before test evaluation; do not tune using these scores.",
            "metrics": metrics,
        }
        (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        with (folder / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction) + "\n")
        with (folder / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_class_id/predicted_class_id", *range(checkpoint["num_classes"])])
            writer.writerows([index, *row] for index, row in enumerate(matrix.tolist()))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
        chart = axis.imshow(matrix.numpy(), cmap="Blues", interpolation="nearest")
        axis.set(xlabel="Predicted class ID", ylabel="True class ID", title=f"Official test — accuracy {metrics['accuracy']:.2%}")
        fig.colorbar(chart, ax=axis, label="Images")
        fig.savefig(folder / "confusion_matrix.png", dpi=160)
        plt.close(fig)
        for key in ("checkpoint_sha256", "source_run_id", "partition", "device"):
            client.log_param(run_id, key, report[key])
        for key, value in metrics.items():
            if key != "per_class":
                client.log_metric(run_id, f"test.{key}", value)
        for path in folder.iterdir():
            client.log_artifact(run_id, str(path), artifact_path="evaluation")
    except BaseException as error:
        client.set_terminated(run_id, status="KILLED" if isinstance(error, KeyboardInterrupt) else "FAILED")
        raise
    else:
        client.set_terminated(run_id, status="FINISHED")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_class"}, indent=2), flush=True)
    print(f"Report: {folder / 'report.json'}", flush=True)
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resnet50/evaluation"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for path in args.checkpoint:
        evaluate_checkpoint(path, device=torch.device(args.device), output_root=args.output_dir)


if __name__ == "__main__":
    main()
