import subprocess
import sys
import pytest
import json
import os
from pathlib import Path
from src import cli


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "tune" in result.stdout
    assert "gradcam" in result.stdout
    assert "predict" in result.stdout
    assert "evaluate" in result.stdout
    assert "test" in result.stdout


def test_cli_subcommand_help():
    for subcmd in ("train", "tune", "gradcam", "predict", "evaluate", "test"):
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", subcmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert f"cli.py {subcmd}" in result.stdout


ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('command',['export-classifier','predict','evaluate','gradcam'])
def test_delivery_commands_have_real_help(command):
    result=subprocess.run([sys.executable,'-B','-m','src.cli',command,'--help'],cwd=ROOT,
        capture_output=True,text=True,timeout=40)
    assert result.returncode==0 and 'usage:' in result.stdout


def test_predict_dispatches_to_shared_library(monkeypatch,capsys):
    from src import cli
    classifier=object()
    monkeypatch.setattr('src.evaluation.inference.load_classifier',lambda path,**kwargs:classifier)
    def predict(loaded,path,**kwargs):
        assert loaded is classifier and Path(path)==Path('bird.png') and kwargs=={'top_k':3}
        return {'class_id':7,'class_name':'bird'}
    monkeypatch.setattr('src.evaluation.inference.predict_image',predict)
    assert cli.main(['predict','--bundle','bundle.json','--image','bird.png','--top-k','3'])==0
    assert json.loads(capsys.readouterr().out)['class_id']==7


def test_evaluate_test_flag_is_required_before_loading_models(monkeypatch,capsys):
    from src import cli
    monkeypatch.setattr('src.evaluation.inference.load_classifier',lambda *a,**k:pytest.fail('model must not load'))
    assert cli.main(['evaluate','--bundle','missing.json','--prepared-dir','prepared',
        '--partition','test','--output-dir','report'])==1
    assert 'official' in capsys.readouterr().err.lower()


def test_evaluate_forwards_canonical_partition_and_options(monkeypatch,tmp_path,capsys):
    from src import cli
    classifier=object()
    monkeypatch.setattr('src.evaluation.inference.load_classifier',lambda *a,**k:classifier)
    def evaluate(loaded,path,**kwargs):
        assert loaded is classifier and Path(path)==Path('prepared')
        assert kwargs=={'partition':'val','batch_size':8,'num_workers':0,'allow_test':False,'output_dir':tmp_path/'evaluation'}
        return {'status':'completed','partition':'val','num_samples':1200,'top1_accuracy':.9}
    monkeypatch.setattr('src.evaluation.inference.evaluate_classifier',evaluate)
    assert cli.main(['evaluate','--bundle','bundle.json','--prepared-dir','prepared','--output-dir',str(tmp_path/'evaluation')])==0
    assert json.loads(capsys.readouterr().out)['num_samples']==1200


def test_delivery_runtime_preserves_measured_cuda_numerical_profile(monkeypatch):
    import torch
    from src import cli
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        cli._delivery_runtime()
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cudnn.benchmark = previous[3]
        torch.backends.cudnn.deterministic = previous[4]
        torch.backends.cudnn.allow_tf32 = previous[5]
        torch.backends.cuda.matmul.allow_tf32 = previous[6]


def _run_training_cli(*args):
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = environment['MKL_NUM_THREADS'] = '2'
    return subprocess.run([sys.executable, '-m', 'src.cli', *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, env=environment, timeout=45)


def test_module_help_exits_successfully_without_data_or_weights():
    result = _run_training_cli('train', '--help')
    assert result.returncode == 0
    assert '--prepared-dir' in result.stdout
    assert '--tracking' in result.stdout


@pytest.mark.parametrize('args', [[], ['train'], ['train', '--task', 'detection']])
def test_parser_errors_exit_nonzero(args):
    result = _run_training_cli(*args)
    assert result.returncode == 2
    assert 'error' in result.stderr.lower()


def test_invalid_json_config_exits_cleanly_without_fetch(tmp_path):
    config = tmp_path / 'invalid.json'
    config.write_text(json.dumps({'architecture': 'missing'}))
    result = _run_training_cli('train', '--task', 'classification', '--config', config,
                               '--prepared-dir', tmp_path / 'absent', '--run-dir', tmp_path / 'run', '--device', 'cpu', '--tracking', 'json')
    assert result.returncode == 1
    assert 'architecture' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not (tmp_path / 'run').exists()


def test_cli_invokes_shared_entry_point_and_returns_summary(monkeypatch, tmp_path, capsys):
    from src import cli as module
    def training(prepared_dir, run_dir, config, **kwargs):
        assert Path(prepared_dir) == tmp_path / 'prepared'
        assert Path(run_dir) == tmp_path / 'run'
        assert config == {'amp': 'off'}
        assert kwargs == {'device': 'cpu', 'tracking': 'json', 'resume': True}
        return {'status': 'completed', 'optimizer_updates': 3}
    monkeypatch.setattr('src.training.experiment.run_training', training)
    config = tmp_path / 'config.json'
    config.write_text('{"amp":"off"}')
    assert module.main(['train', '--task', 'classification', '--config', str(config),
                        '--prepared-dir', str(tmp_path / 'prepared'), '--run-dir', str(tmp_path / 'run'),
                        '--device', 'cpu', '--tracking', 'json', '--resume']) == 0
    assert json.loads(capsys.readouterr().out)['optimizer_updates'] == 3


def test_cli_forwards_initial_checkpoint_to_shared_controller(monkeypatch, tmp_path, capsys):
    from src import cli as module
    received = {}
    def training(prepared_dir, run_dir, config, **kwargs):
        received.update(kwargs)
        return {'status': 'completed'}
    monkeypatch.setattr('src.training.experiment.run_training', training)
    config = tmp_path / 'config.json'
    config.write_text('{}')
    parent = tmp_path / 'parent.pt'
    assert module.main(['train', '--task', 'classification', '--config', str(config),
                        '--prepared-dir', str(tmp_path / 'prepared'), '--run-dir', str(tmp_path / 'run'),
                        '--device', 'cpu', '--tracking', 'json', '--initial-checkpoint', str(parent)]) == 0
    assert received['initial_checkpoint'] == parent
    assert received['resume'] is False


def test_cli_rejects_initial_checkpoint_with_resume_before_reading_config(tmp_path):
    result = _run_training_cli('train', '--task', 'classification', '--config', tmp_path / 'missing.json',
                               '--prepared-dir', tmp_path / 'prepared', '--run-dir', tmp_path / 'run',
                               '--initial-checkpoint', tmp_path / 'parent.pt', '--resume')
    assert result.returncode == 2
    assert 'not allowed with argument' in result.stderr


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
    from src.training import scratch_trainer, trainer

    phase = trainer.Phase("partial", 1, 0.01, backbone_lr=0.001)
    config = trainer.TrainConfig(phases=(phase,), weights=None, selection_metric="val_macro_f1",
                                 label_smoothing=0.1, color_jitter=0.2, mixup_alpha=0.2)
    assert package.TrainConfig is trainer.TrainConfig
    assert package.Phase is trainer.Phase
    assert package.fit is trainer.fit
    assert package.TrainConfig.from_dict(config.to_dict()) == config
    assert trainer.TrainingConfig is scratch_trainer.TrainingConfig
    assert trainer.train_classification is scratch_trainer.train_classification
