"""Compare equal-budget grid and random LDA shrinkage searches on cached features."""

import argparse
import csv
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from src.data.preparation import load_classification_manifest
from src.resnet50.evaluate import metrics_from_confusion
from src.resnet50.head_refit import fit_discriminant_head


ROOT = Path(__file__).resolve().parents[2]
LOWER, UPPER = 0.03, 0.95


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def candidates(trials: int, seed: int) -> dict[str, list[float]]:
    """Sample log-spaced grid points and log-uniform random points in [LOWER, UPPER]."""
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 2:
        raise ValueError("trials must be an integer >= 2.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    rng = np.random.default_rng(seed)
    return {
        "grid": np.geomspace(LOWER, UPPER, trials).tolist(),
        "random": np.exp(rng.uniform(np.log(LOWER), np.log(UPPER), trials)).tolist(),
    }


def load_features(features_path: Path, source_path: Path) -> tuple[dict, dict]:
    """Load cached train/val features and verify hashes and row ordering against the checkpoint."""
    cache = torch.load(features_path, weights_only=True, map_location="cpu", mmap=True)
    if set(cache["partitions"]) != {"train", "val"} or set(cache["image_ids"]) != {"train", "val"}:
        raise ValueError("Feature cache must contain train and val only, with image identities.")
    source_hash = sha256(source_path)
    if cache["source_sha256"] != source_hash:
        raise ValueError("Feature cache belongs to a different encoder checkpoint.")
    source = torch.load(source_path, weights_only=True, map_location="cpu", mmap=True)
    config = source["config"]
    prepared = Path(config["data"]["prepared_dir"])
    if not prepared.is_absolute():
        prepared = ROOT / prepared
    rows = load_classification_manifest(prepared)
    preparation = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    for key in ("manifest_sha256", "class_names", "source_revisions"):
        if preparation[key] != config["data_provenance"][key]:
            raise ValueError(f"Encoder and preparation provenance differ: {key}.")
    for partition in ("train", "val"):
        canonical = sorted((row for row in rows if row["partition"] == partition), key=lambda row: row["source_row"])
        x, y = cache["partitions"][partition]
        ids = cache["image_ids"][partition]
        if ids != [row["image_id"] for row in canonical] or y.tolist() != [row["class_id"] for row in canonical]:
            raise ValueError(f"Cached {partition} image identities/labels differ from preparation.")
        if x.ndim != 2 or len(x) != len(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid {partition} feature population.")
    if set(cache["image_ids"]["train"]) & set(cache["image_ids"]["val"]):
        raise ValueError("Training and validation images overlap.")
    metadata = {
        "source_checkpoint": str(source_path.resolve()), "source_sha256": source_hash,
        "features": str(features_path.resolve()), "features_sha256": sha256(features_path),
        "num_classes": source["num_classes"], "feature_dimensions": cache["partitions"]["train"][0].shape[1],
        "train_samples": len(cache["partitions"]["train"][1]), "val_samples": len(cache["partitions"]["val"][1]),
        "manifest_sha256": preparation["manifest_sha256"],
    }
    return cache["partitions"], metadata


def run_trials(features: dict, settings: dict, num_classes: int):
    """Interleave grid and random LDA fits, timing fit + validation scoring per trial."""
    if set(settings) != {"grid", "random"} or len(settings["grid"]) != len(settings["random"]):
        raise ValueError("Grid and random must have the same trial budget.")
    train_x, train_y = features["train"]
    val_x, val_y = features["val"]
    for index in range(len(settings["grid"])):
        # Alternate pair order each iteration to reduce ordering effects on CPU timing.
        order = ("grid", "random") if index % 2 == 0 else ("random", "grid")
        for method in order:
            value = settings[method][index]
            started = perf_counter()
            fit = fit_discriminant_head(train_x, train_y, val_x, val_y,
                num_classes=num_classes, shrinkage=value, device=torch.device("cpu"), calibrate=False)
            logits = F.linear(val_x.float(), fit.weight, fit.bias)
            guessed = logits.argmax(1)
            matrix = torch.bincount(val_y * num_classes + guessed, minlength=num_classes**2).reshape(num_classes, num_classes)
            values = metrics_from_confusion(matrix)
            row = {"method": method, "trial": index + 1, "shrinkage": value,
                   "seconds": perf_counter() - started, "train_accuracy": fit.train_metrics["accuracy"],
                   **{f"val_{key}": values[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}}
            yield row, fit


def summarize(rows: list[dict]) -> dict:
    result = {}
    for method in ("grid", "random"):
        trials = [row for row in rows if row["method"] == method]
        best = max(trials, key=lambda row: (row["val_accuracy"], row["val_macro_f1"], -row["trial"]))
        result[method] = {"trials": len(trials), "fit_score_seconds": sum(row["seconds"] for row in trials), "best": best}
    return result


def plot_progress(rows: list[dict], destination: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 3.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for method in ("grid", "random"):
        trials = [row for row in rows if row["method"] == method]
        best = np.maximum.accumulate([row["val_accuracy"] for row in trials]) * 100
        seconds = np.cumsum([row["seconds"] for row in trials])
        axes[0].plot(range(1, len(trials) + 1), best, marker="o", label=method)
        axes[1].plot(seconds, best, marker="o", label=method)
    for axis in axes:
        axis.set_ylabel("Best validation accuracy (%)")
        axis.legend()
        axis.grid(alpha=0.25)
    axes[0].set_xlabel("Completed trials")
    axes[1].set_xlabel("Cumulative fit + score seconds")
    figure.savefig(destination, dpi=150)
    figure.clear()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument("--trials", type=int, default=8, help="Trials per method (default: 8).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory.")
    try:
        settings = candidates(args.trials, args.seed)
        torch.set_num_threads(2)
        features, metadata = load_features(args.features, args.source_checkpoint)
        # Warm up torch linear algebra before timing trials.
        torch.linalg.solve(torch.eye(32, dtype=torch.float64), torch.ones(32, 1, dtype=torch.float64))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        plan = {**metadata, "settings": settings, "seed": args.seed, "threads": 2,
                "bounds": [LOWER, UPPER], "scale": "log", "device": "cpu", "calibration": False,
                "selection": "Validation accuracy, then macroF1, then earlier trial",
                "timing": "Fit + validation scoring only; excludes feature loading/extraction, logging and saving",
                "test_evaluated": False, "selected_model_replaced": False}
        (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        import mlflow

        tracking = ROOT / "outputs/resnet50/mlruns"
        tracking.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
        if mlflow.get_experiment_by_name("resnet50_grid_vs_random") is None:
            mlflow.create_experiment("resnet50_grid_vs_random", artifact_location=(tracking / "artifacts").as_uri())
        mlflow.set_experiment("resnet50_grid_vs_random")
        rows = []
        heads = args.output_dir / "heads"
        heads.mkdir()
        with mlflow.start_run(run_name=f"equal_budget_seed_{args.seed}") as parent:
            mlflow.log_params({key: value for key, value in plan.items() if key != "settings"})
            for row, fit in run_trials(features, settings, metadata["num_classes"]):
                with mlflow.start_run(run_name=f"{row['method']}_{row['trial']:02d}", nested=True) as trial:
                    mlflow.log_params({"method": row["method"], "shrinkage": row["shrinkage"], "source_sha256": metadata["source_sha256"]})
                    mlflow.log_metrics({key: value for key, value in row.items() if key not in ("method", "trial", "shrinkage")})
                    head_path = heads / f"{row['method']}_{row['trial']:02d}.pt"
                    torch.save({"weight": fit.weight, "bias": fit.bias, "source_sha256": metadata["source_sha256"],
                                "shrinkage": row["shrinkage"], "feature_cache_sha256": metadata["features_sha256"]}, head_path)
                    mlflow.log_artifact(str(head_path), artifact_path="classifier")
                    row["run_id"] = trial.info.run_id
                rows.append(row)
                print(f"{row['method']} trial {row['trial']}/{args.trials}: shrinkage={row['shrinkage']:.5f}, "
                      f"val accuracy={row['val_accuracy']:.2%}, F1={row['val_macro_f1']:.2%}", flush=True)
            for path, key in ((args.features, "features_sha256"), (args.source_checkpoint, "source_sha256")):
                if sha256(path) != metadata[key]:
                    raise RuntimeError("An input changed during the comparison.")
            summary = {"methods": summarize(rows), "parent_run_id": parent.info.run_id,
                       "test_evaluated": False, "selected_model_replaced": False,
                       "limitations": "One encoder, one parameter and one random seed; timing is machine-dependent. Validation was reused."}
            with (args.output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            plot_progress(rows, args.output_dir / "comparison.png")
            for name in ("plan.json", "trials.csv", "summary.json", "comparison.png"):
                mlflow.log_artifact(str(args.output_dir / name), artifact_path="comparison")
            mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="code")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Results: {args.output_dir.resolve()}", flush=True)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f"Search comparison failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
