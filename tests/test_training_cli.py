"""Execute the public module parser rather than inspecting CLI source text."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = environment['MKL_NUM_THREADS'] = '2'
    return subprocess.run([sys.executable, '-m', 'src.cli', *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, env=environment, timeout=45)


def test_module_help_exits_successfully_without_data_or_weights():
    result = cli('train', '--help')
    assert result.returncode == 0
    assert '--prepared-dir' in result.stdout
    assert '--tracking' in result.stdout


@pytest.mark.parametrize('args', [[], ['train'], ['train', '--task', 'detection']])
def test_parser_errors_exit_nonzero(args):
    result = cli(*args)
    assert result.returncode == 2
    assert 'error' in result.stderr.lower()


def test_invalid_json_config_exits_cleanly_without_fetch(tmp_path):
    config = tmp_path / 'invalid.json'
    config.write_text(json.dumps({'architecture': 'missing'}))
    result = cli('train', '--task', 'classification', '--config', config,
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
