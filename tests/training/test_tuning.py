"""Offline unit tests for hyperparameter tuning (Grid Search and Random Search)."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.tuning import compare_tuning, grid_search, random_search


def make_test_loaders():
    torch.manual_seed(42)
    x_train = torch.randn(8, 3, 32, 32)
    y_train = torch.randint(0, 200, (8,))
    x_val = torch.randn(4, 3, 32, 32)
    y_val = torch.randint(0, 200, (4,))

    def collate(batch):
        return {
            "image": torch.stack([b[0] for b in batch]),
            "label": torch.tensor([b[1] for b in batch]),
        }

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=4, collate_fn=collate
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=4, collate_fn=collate
    )
    return {"train": train_loader, "val": val_loader}


def test_grid_search_evaluates_all_combinations():
    loaders = make_test_loaders()
    grid = {
        "optimizer": ["adam", "sgd"],
        "activation": ["relu"],
        "learning_rate": [1e-3],
    }
    result = grid_search(grid, loaders, epochs=1, device="cpu", seed=10)
    assert result.num_trials == 2
    assert len(result.trials) == 2
    assert result.method == "grid_search"
    assert result.best_trial.best_val_accuracy >= 0.0
    assert result.total_time_seconds > 0


def test_random_search_samples_specified_number_of_trials():
    loaders = make_test_loaders()
    distributions = {
        "optimizer": ["adam", "sgd", "adagrad"],
        "activation": ["relu", "leaky_relu"],
        "learning_rate": [1e-3, 3e-4],
    }
    result = random_search(distributions, loaders, num_trials=3, epochs=1, device="cpu", seed=42)
    assert result.num_trials == 3
    assert len(result.trials) == 3
    assert result.method == "random_search"
    assert result.best_trial.best_val_accuracy >= 0.0


def test_compare_tuning_produces_summary():
    loaders = make_test_loaders()
    grid = {"optimizer": ["adam"], "learning_rate": [1e-3]}
    res_grid = grid_search(grid, loaders, epochs=1, device="cpu", seed=1)
    res_rand = random_search(grid, loaders, num_trials=1, epochs=1, device="cpu", seed=2)

    comp = compare_tuning(res_grid, res_rand)
    assert "grid_search" in comp
    assert "random_search" in comp
    assert "comparison" in comp
    assert "accuracy_difference" in comp["comparison"]
    assert "time_ratio_random_over_grid" in comp["comparison"]
