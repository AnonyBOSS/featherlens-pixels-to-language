"""Offline regressions for scratch source selection and initialization seeds."""
from argparse import Namespace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src import cli
from src.data.source import _SOURCES
from src.training import tuning


class MetadataOnlyRows:
    """Dataset construction may inspect length, but evaluation is stubbed."""

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError("Source pixels must never be read in this test")


@pytest.mark.parametrize("partition", ["train", "val", "test"])
def test_scratch_evaluation_uses_matching_pinned_source(monkeypatch, tmp_path, partition):
    names = tuple(f"Bird {index}" for index in range(200))
    sources = {
        split: SimpleNamespace(spec=spec, dataset=MetadataOnlyRows(), class_names=names)
        for split, spec in _SOURCES.items()
    }
    rows = []
    for split_name in ("train", "val", "test"):
        source_split = "test" if split_name == "test" else "train"
        spec = _SOURCES[source_split]
        rows.append({
            "partition": split_name, "source_row": 0, "class_id": 0,
            "source_split": source_split, "source_repo": spec.repo_id,
            "source_revision": spec.revision, "image_id": spec.image_id(0),
            "eligible": True, "class_name": names[0],
        })
    model = nn.Linear(3, 200)
    checkpoint = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    config = tmp_path / "scratch.yaml"
    config.write_text(f"output_dir: {tmp_path.as_posix()}\ndata:\n  num_workers: 0\n", encoding="utf-8")
    monkeypatch.setattr(cli, "create_scratch_model", lambda config: nn.Linear(3, 200))
    monkeypatch.setattr(cli, "load_classification_manifest", lambda path: rows)
    monkeypatch.setattr(cli, "load_cub_split", lambda split: sources[split])
    monkeypatch.setattr(cli, "evaluate_classification", lambda *args, **kwargs: (0.5, torch.eye(200, dtype=torch.int64)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cli._cmd_evaluate(Namespace(config=config, checkpoint=checkpoint, split=partition, allow_official_test=True))

    metrics = json.loads((tmp_path / f"{partition}_metrics.json").read_text(encoding="utf-8"))
    assert metrics["num_samples"] == 1
    assert metrics["loss"] == 0.5


@pytest.mark.parametrize("legacy_namespace", [False, True])
def test_scratch_cli_seed_controls_initial_model_weights(monkeypatch, tmp_path, legacy_namespace):
    config = tmp_path / "scratch.yaml"
    config.write_text(f"output_dir: {tmp_path.as_posix()}\ntraining:\n  seed: 123\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_loaders_from_config", lambda config: ({}, None))
    monkeypatch.setattr(cli, "create_scratch_model", lambda config: nn.Linear(3, 2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    initial_weights = []

    class TrainingBoundary(Exception):
        pass

    def capture_initial_weights(model, loaders, config, **kwargs):
        initial_weights.append(model.weight.detach().clone())
        raise TrainingBoundary

    monkeypatch.setattr(cli, "train_classification", capture_initial_weights)
    args = (Namespace(config=config) if legacy_namespace
            else Namespace(config=config, run_dir=None, prepared_dir=None, device="cpu"))
    for ambient_seed in (7, 987):
        torch.manual_seed(ambient_seed)
        with pytest.raises(TrainingBoundary):
            cli._cmd_train(args)
    assert torch.equal(initial_weights[0], initial_weights[1])


def test_tuning_trial_seed_controls_initial_model_weights(monkeypatch):
    monkeypatch.setattr(tuning, "create_scratch_model", lambda config: nn.Linear(3, 2))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    initial_weights = []

    def capture_initial_weights(model, loaders, config, **kwargs):
        initial_weights.append(model.weight.detach().clone())
        return SimpleNamespace(best_val_accuracy=0.0, best_epoch=1)

    monkeypatch.setattr(tuning, "train_classification", capture_initial_weights)
    for ambient_seed in (7, 987):
        torch.manual_seed(ambient_seed)
        tuning._run_trial(1, {}, {}, epochs=1, device="cpu", seed=123)
    tuning._run_trial(2, {}, {}, epochs=1, device="cpu", seed=124)
    assert torch.equal(initial_weights[0], initial_weights[1])
    assert not torch.equal(initial_weights[0], initial_weights[2])
