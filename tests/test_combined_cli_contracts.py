"""Shared CLI formats and canonical trainer compatibility, without model inference."""
import json
from pathlib import Path

import pytest

from src import cli


@pytest.mark.parametrize("command,extra", [
    ("predict", ["--image", "bird.png"]), ("evaluate", []), ("gradcam", []),
])
def test_cli_rejects_ambiguous_checkpoint_and_bundle(command, extra, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--checkpoint", "scratch.pt", "--bundle", "bundle", *extra])
    assert error.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["evaluate", "test"])
@pytest.mark.parametrize("bundle", [False, True])
def test_official_test_requires_opt_in_before_loading(command, bundle, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model or source may be opened before official-test opt-in")
    monkeypatch.setattr(cli, "_cmd_bundle", forbidden)
    monkeypatch.setattr(cli, "_cmd_evaluate", forbidden)
    args = [command, "--split", "test"]
    if bundle:
        args += ["--bundle", "classifier", "--prepared-dir", "prepared", "--output-dir", "report"]
    assert cli.main(args) == 1
    assert "--allow-official-test" in capsys.readouterr().err


def test_bundle_and_legacy_checkpoint_prediction_dispatch(monkeypatch, capsys):
    def bundle(args):
        assert args.bundle == Path("classifier")
        assert args.checkpoint is None
        assert args.image == Path("bird.png")
        assert args.top_k == 3
        return {"class_id": 7, "class_name": "Bird 7"}

    def scratch(args):
        assert args.checkpoint == Path("scratch.pt")
        assert args.bundle is None
        print("Prediction: Bird 9 (Class ID: 9)")

    monkeypatch.setattr(cli, "_cmd_bundle", bundle)
    monkeypatch.setattr(cli, "_cmd_predict", scratch)
    assert cli.main(["predict", "--bundle", "classifier", "--image", "bird.png", "--top-k", "3"]) == 0
    assert json.loads(capsys.readouterr().out) == {"class_id": 7, "class_name": "Bird 7"}
    assert cli.main(["predict", "--checkpoint", "scratch.pt", "--image", "bird.png"]) == 0
    assert "Class ID: 9" in capsys.readouterr().out


def test_pretrained_initial_checkpoint_is_forwarded(monkeypatch, tmp_path, capsys):
    config = tmp_path / "pretrained.json"
    config.write_text("{}", encoding="utf-8")

    def training(prepared, run, values, **options):
        assert prepared == Path("prepared") and run == Path("run")
        assert values == {}
        assert options == {"device": "cpu", "resume": False, "tracking": "json",
                           "initial_checkpoint": Path("initial.pt")}
        return {"status": "completed"}

    monkeypatch.setattr("src.training.experiment.run_training", training)
    assert cli.main(["train", "--config", str(config), "--prepared-dir", "prepared",
                     "--run-dir", "run", "--initial-checkpoint", "initial.pt",
                     "--device", "cpu", "--tracking", "json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "completed"}


def test_public_pretrained_imports_share_updated_config_and_fit():
    import src.training as package
    from src.training import pretrained_trainer as compatibility
    from src.training import scratch_trainer, trainer

    phase = trainer.Phase("partial", 1, 0.01, backbone_lr=0.001)
    config = trainer.TrainConfig(phases=(phase,), weights=None, selection_metric="val_macro_f1",
                                 label_smoothing=0.1, color_jitter=0.2, mixup_alpha=0.2)
    assert compatibility.TrainConfig is package.TrainConfig is trainer.TrainConfig
    assert compatibility.Phase is package.Phase is trainer.Phase
    assert compatibility.fit is package.fit is trainer.fit
    assert compatibility.TrainConfig.from_dict(config.to_dict()) == config
    assert trainer.TrainingConfig is scratch_trainer.TrainingConfig
    assert trainer.train_classification is scratch_trainer.train_classification
