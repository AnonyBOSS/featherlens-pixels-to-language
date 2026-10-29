"""Simple Grid Search and Random Search hyperparameter tuning for classification."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import itertools
import random
import time
from typing import Any

from torch import nn
from torch.utils.data import DataLoader

from src.models.scratch import ScratchConfig, create_scratch_model
from src.training.trainer import TrainingConfig, train_classification


@dataclass(frozen=True)
class TrialResult:
    """Outcome and metadata of one hyperparameter evaluation trial."""

    trial_id: int
    params: dict[str, Any]
    best_val_accuracy: float
    best_epoch: int
    duration_seconds: float


@dataclass(frozen=True)
class TuningResult:
    """Aggregated results across all trials of a tuning run."""

    method: str
    trials: tuple[TrialResult, ...]
    best_trial: TrialResult
    total_time_seconds: float

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "num_trials": self.num_trials,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "best_params": self.best_trial.params,
            "best_val_accuracy": round(self.best_trial.best_val_accuracy, 4),
            "best_epoch": self.best_trial.best_epoch,
        }


def _run_trial(
    trial_id: int,
    params: dict[str, Any],
    loaders: Mapping[str, DataLoader],
    epochs: int,
    device: str | None = None,
    seed: int = 42,
) -> TrialResult:
    """Instantiate model and train for specified epochs under params."""
    model_config = ScratchConfig(
        activation=params.get("activation", "relu"),
        head_dropout=params.get("head_dropout", 0.0),
    )
    model = create_scratch_model(model_config)

    train_config = TrainingConfig(
        epochs=epochs,
        learning_rate=params.get("learning_rate", 3e-4),
        weight_decay=params.get("weight_decay", 1e-4),
        optimizer=params.get("optimizer", "adam"),
        seed=seed,
        amp=False if device == "cpu" else True,
    )

    start_time = time.time()
    result = train_classification(model, loaders, train_config, device=device)
    duration = time.time() - start_time

    return TrialResult(
        trial_id=trial_id,
        params=dict(params),
        best_val_accuracy=result.best_val_accuracy,
        best_epoch=result.best_epoch,
        duration_seconds=duration,
    )


def grid_search(
    param_grid: Mapping[str, Sequence[Any]],
    loaders: Mapping[str, DataLoader],
    *,
    epochs: int = 2,
    device: str | None = None,
    seed: int = 42,
) -> TuningResult:
    """Exhaustively evaluate all Cartesian product combinations in param_grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    trials: list[TrialResult] = []
    start_total = time.time()

    for idx, combo in enumerate(combinations, start=1):
        params = dict(zip(keys, combo))
        trial = _run_trial(idx, params, loaders, epochs=epochs, device=device, seed=seed + idx)
        trials.append(trial)

    total_time = time.time() - start_total
    best_trial = max(trials, key=lambda t: t.best_val_accuracy)
    return TuningResult(
        method="grid_search",
        trials=tuple(trials),
        best_trial=best_trial,
        total_time_seconds=total_time,
    )


def random_search(
    param_distributions: Mapping[str, Sequence[Any]],
    loaders: Mapping[str, DataLoader],
    *,
    num_trials: int = 5,
    epochs: int = 2,
    device: str | None = None,
    seed: int = 42,
) -> TuningResult:
    """Randomly sample parameter configurations from param_distributions."""
    if num_trials < 1:
        raise ValueError("num_trials must be >= 1")

    rng = random.Random(seed)
    trials: list[TrialResult] = []
    start_total = time.time()

    for idx in range(1, num_trials + 1):
        params = {k: rng.choice(v) for k, v in param_distributions.items()}
        trial = _run_trial(idx, params, loaders, epochs=epochs, device=device, seed=seed + idx)
        trials.append(trial)

    total_time = time.time() - start_total
    best_trial = max(trials, key=lambda t: t.best_val_accuracy)
    return TuningResult(
        method="random_search",
        trials=tuple(trials),
        best_trial=best_trial,
        total_time_seconds=total_time,
    )


def compare_tuning(grid: TuningResult, random_res: TuningResult) -> dict[str, Any]:
    """Produce the required comparison table between Grid Search and Random Search."""
    return {
        "grid_search": grid.summary(),
        "random_search": random_res.summary(),
        "comparison": {
            "accuracy_difference": round(
                random_res.best_trial.best_val_accuracy - grid.best_trial.best_val_accuracy, 4
            ),
            "time_ratio_random_over_grid": round(
                random_res.total_time_seconds / max(grid.total_time_seconds, 1e-6), 2
            ),
            "trial_count_ratio": round(random_res.num_trials / max(grid.num_trials, 1), 2),
        },
    }
