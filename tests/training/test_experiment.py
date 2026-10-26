"""Synthetic CPU orchestration; these fixtures are not CUB acceptance evidence."""
from dataclasses import replace
import csv
import hashlib
import importlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.classification import make_classification_loaders
from src.models.pretrained import ModelConfig
from src.training.checkpoint import read_checkpoint
from src.training.runtime import CANONICAL_MANIFEST_SHA256
from src.training.trainer import TrainConfig, Phase


class Samples(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {'image': torch.tensor([index % 7 / 7, index % 3 / 3, 1.0]),
                'label': index % 200, 'image_id': f'image-{index}'}


class TinyAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(3, 3)
        self.head = nn.Sequential(nn.Dropout(config.head_dropout), nn.Linear(3, 200))
        self.set_fine_tuning(config.fine_tuning, freeze_bn_stats=config.freeze_bn_stats)

    def set_fine_tuning(self, mode, *, freeze_bn_stats=True):
        self.config = replace(self.config, fine_tuning=mode, freeze_bn_stats=freeze_bn_stats)
        self.encoder.requires_grad_(mode != 'frozen')
        return self

    def forward(self, images):
        return self.head(self.encoder(images))


@pytest.fixture
def setup(monkeypatch, tmp_path):
    assert importlib.util.find_spec('src.training.experiment') is not None, 'Experiment entry point is missing'
    api = importlib.import_module('src.training.experiment')
    identity = {'status': 'metadata_validated', 'manifest_sha256': CANONICAL_MANIFEST_SHA256,
                'preparation_sha256': 'a' * 64, 'class_mapping_sha256': 'b' * 64,
                'class_mapping_hash_encoding': 'UTF-8 compact JSON ordered class_names, ensure_ascii=False',
                'class_names': [f'class-{i}' for i in range(200)],
                'partition_counts': {'train': 4794, 'val': 1200, 'test': 5794, 'excluded': 0},
                'seed': 42, 'source_revisions': {'train': 'synthetic'},
                'official_test_images_accessed': False, 'source_status': 'not_run'}
    config = TrainConfig(amp='off', batch_size=512, accumulation_steps=2, head_dropout=0.2)
    monkeypatch.setattr(api, 'validate_preflight_preparation', lambda path: identity)

    def loaders(path, model_config, **kwargs):
        assert kwargs['seed'] == 42
        assert kwargs['preprocessing'].image_size == 224
        return make_classification_loaders({'train': Samples(4794), 'val': Samples(1200)},
                                           batch_size=kwargs['batch_size'], num_workers=kwargs['num_workers'], seed=42)

    def builder(model_config):
        return TinyAdapter(model_config), {'status': 'synthetic fixture, no official weights', 'sha256': 'c' * 64}

    monkeypatch.setattr(api, 'make_pretrained_loaders', loaders)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', builder)
    monkeypatch.setattr('src.training.checkpoint.create_pretrained_model',
                        lambda config, *, load_weights: TinyAdapter(config) if load_weights is False else pytest.fail('download'))
    return api, config, identity, tmp_path / 'run'


def run(setup, **kwargs):
    api, config, _, directory = setup
    return api.run_training(directory.parent / 'prepared', directory, config, device='cpu', tracking='json', **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail('Unexpected source, weight or training access')


def test_cpu_training_exports_metrics_predictions_reload_proof_and_light_evidence(setup):
    result = run(setup)
    _, config, identity, directory = setup
    assert result['epochs_completed'] == 2
    assert result['optimizer_updates'] == 10
    assert result['skipped_updates'] == 0
    assert result['status'] == 'completed'
    assert result['data_identity'] == identity
    assert result['checkpoint_reload']['status'] == 'forward_verified'
    assert result['checkpoint_reload']['logits_shape'] == [512, 200]
    assert result['checkpoint_reload']['finite'] is True
    assert result['tracking']['mode'] == 'json'
    assert result['weight_provenance']['status'].startswith('synthetic')
    assert json.loads((directory / 'config.json').read_text()) == config.to_dict()
    history = json.loads((directory / 'history.json').read_text())
    assert [(r['num_train_samples'], r['num_val_samples']) for r in history] == [(4794, 1200)] * 2
    with (directory / 'history.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 2
    for suffix in ('best', 'latest'):
        report = json.loads((directory / f'validation-{suffix}.json').read_text())
        assert 'predictions' not in report
        assert len(report['per_class']) == 200
        with (directory / f'predictions-{suffix}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1200
        assert len({row['image_id'] for row in rows}) == 1200
        checkpoint = result['checkpoints'][suffix]
        assert checkpoint['path'] == f'{suffix}.pt'
        assert checkpoint['sha256'] == hashlib.sha256((directory / checkpoint['path']).read_bytes()).hexdigest()
    with ZipFile(directory / 'evidence.zip') as archive:
        names = set(archive.namelist())
        assert {'config.json', 'run_summary.json', 'history.csv', 'sha256.json', 'run.log'} <= names
        assert not any('.pt' in name or 'mlruns' in name for name in names)
        inventory = json.loads(archive.read('sha256.json'))
        for name, digest in inventory.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_fresh_seed_reproduces_random_head_and_training(setup):
    first = run(setup)
    api, config, _, directory = setup
    second = api.run_training(directory.parent / 'prepared', directory.parent / 'other', config,
                              device='cpu', tracking='json')
    assert first['history'] == second['history']
    a, b = read_checkpoint(directory / 'latest.pt'), read_checkpoint(directory.parent / 'other/latest.pt')
    for key in a['model_state']:
        assert torch.equal(a['model_state'][key], b['model_state'][key])


@pytest.mark.parametrize('remove_summary', [False, True])
def test_terminal_resume_rebuilds_without_preparation_pixels_weights_or_fit(setup, monkeypatch, remove_summary):
    initial = run(setup)
    api, _, _, directory = setup
    digest = hashlib.sha256((directory / 'latest.pt').read_bytes()).hexdigest()
    if remove_summary:
        (directory / 'run_summary.json').unlink()
    (directory / 'history.csv').unlink()
    for name in ('validate_preflight_preparation', 'make_pretrained_loaders', 'build_verified_pretrained_model', 'fit'):
        monkeypatch.setattr(api, name, forbidden)
    result = run(setup, resume=True)
    assert result['history'] == initial['history']
    assert result['optimizer_updates'] == 10
    assert (directory / 'history.csv').is_file()
    assert hashlib.sha256((directory / 'latest.pt').read_bytes()).hexdigest() == digest
    assert result['checkpoint_reload']['status'] == ('state_loaded_without_forward' if remove_summary else 'forward_verified')


def test_nonempty_fresh_directory_is_rejected_before_source_or_weights(setup, monkeypatch):
    api, _, _, directory = setup
    directory.mkdir()
    (directory / 'keep.txt').write_text('existing output')
    monkeypatch.setattr(api, 'validate_preflight_preparation', forbidden)
    with pytest.raises(FileExistsError, match='resume|fresh'):
        run(setup)
    assert (directory / 'keep.txt').read_text() == 'existing output'


@pytest.mark.parametrize('change', [{'seed': 7}, {'amp': 'float16'}, {'architecture': 'missing'}])
def test_invalid_config_has_no_source_or_weight_side_effects(setup, monkeypatch, change):
    api, config, _, directory = setup
    for name in ('validate_preflight_preparation', 'make_pretrained_loaders', 'build_verified_pretrained_model'):
        monkeypatch.setattr(api, name, forbidden)
    with pytest.raises((ValueError, TypeError)):
        api.run_training('missing', directory, config.to_dict() | change, device='cpu', tracking='json')
    assert not directory.exists()


def test_invalid_preparation_does_not_open_source_or_fetch_weights(setup, monkeypatch):
    api, _, _, _ = setup
    def invalid(path):
        raise ValueError('canonical metadata invalid')
    monkeypatch.setattr(api, 'validate_preflight_preparation', invalid)
    monkeypatch.setattr(api, 'make_pretrained_loaders', forbidden)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    with pytest.raises(ValueError, match='canonical metadata'):
        run(setup)


def test_wrong_loader_lengths_fail_before_official_weights(setup, monkeypatch):
    api, _, _, _ = setup
    monkeypatch.setattr(api, 'make_pretrained_loaders', lambda *args, **kwargs:
                        make_classification_loaders({'train': Samples(3), 'val': Samples(2)}, batch_size=512))
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    with pytest.raises(ValueError, match='4794|1200|length'):
        run(setup)


def test_failure_preserves_completed_epoch_and_resume_uses_original_provenance(setup, monkeypatch):
    import src.training.trainer as trainer
    real_train = trainer.train_epoch
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('synthetic interruption')
        return real_train(*args, **kwargs)
    monkeypatch.setattr(trainer, 'train_epoch', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(setup)
    api, _, _, directory = setup
    state = json.loads((directory / 'run_state.json').read_text())
    assert state['status'] == 'interrupted'
    assert read_checkpoint(directory / 'latest.pt')['epochs_completed'] == 1
    assert len(json.loads((directory / 'history.json').read_text())) == 1
    monkeypatch.setattr(trainer, 'train_epoch', real_train)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    result = run(setup, resume=True)
    assert result['epochs_completed'] == 2
    assert result['optimizer_updates'] == 10
    assert result['weight_provenance']['status'].startswith('synthetic')
    assert result['failures'][0]['type'] == 'KeyboardInterrupt'


def test_resume_rejects_changed_config_and_corruption_before_loaders(setup, monkeypatch):
    run(setup)
    api, config, _, directory = setup
    monkeypatch.setattr(api, 'make_pretrained_loaders', forbidden)
    with pytest.raises(ValueError, match='config'):
        api.run_training('missing', directory, replace(config, accumulation_steps=1),
                         device='cpu', tracking='json', resume=True)
    (directory / 'latest.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        run(setup, resume=True)


def test_real_local_mlflow_run_reopens_same_id_and_reads_metrics_artifacts(setup):
    pytest.importorskip('mlflow')
    api, config, _, directory = setup
    result = api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=(directory / 'mlruns').resolve().as_uri())
    run_id = result['tracking']['run_id']
    assert result['tracking']['status'] == 'finished'
    tracked = client.get_run(run_id)
    assert tracked.info.status == 'FINISHED'
    assert tracked.data.params['architecture'] == 'convnext_tiny'
    metrics = client.get_metric_history(run_id, 'val_loss')
    assert [(m.step, m.value) for m in metrics] == [(r['epoch'], r['val_loss']) for r in result['history']]
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert {'run_summary.json', 'history.csv', 'config.json'} <= artifacts
    assert not any('.pt' in name for name in artifacts)
    downloaded = client.download_artifacts(run_id, 'history.csv', str(directory.parent))
    assert Path(downloaded).read_bytes() == (directory / 'history.csv').read_bytes()
    again = api.run_training('missing', directory, config, device='cpu', tracking='mlflow', resume=True)
    assert again['tracking']['run_id'] == run_id
    assert len(client.get_metric_history(run_id, 'val_loss')) == 2


def test_requested_missing_mlflow_fails_before_model_or_updates(setup, monkeypatch):
    import sys
    api, config, _, directory = setup
    monkeypatch.setitem(sys.modules, 'mlflow', None)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    with pytest.raises(ImportError, match='mlflow-skinny==3.17.2'):
        api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    assert not (directory / 'latest.pt').exists()


def test_mlflow_logging_failure_is_visible_and_does_not_discard_training(setup, monkeypatch, capsys):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic tracking failure')
    monkeypatch.setattr(MlflowClient, 'log_metric', fail)
    api, config, _, directory = setup
    result = api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    assert result['epochs_completed'] == 2
    assert result['tracking']['status'] == 'error'
    assert any(item['stage'] == 'tracking' for item in result['failures'])
    assert 'synthetic tracking failure' in capsys.readouterr().err
    assert (directory / 'latest.pt').is_file()
    assert json.loads((directory / 'run_summary.json').read_text())['tracking']['status'] == 'error'


def test_nonterminal_resume_validates_model_tensors_before_opening_source(setup, monkeypatch):
    import src.training.trainer as trainer
    from src.training.checkpoint import save_checkpoint
    original = trainer.train_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('stop after completed epoch')
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, 'train_epoch', interrupt)
    with pytest.raises(RuntimeError):
        run(setup)
    api, _, _, directory = setup
    payload = read_checkpoint(directory / 'latest.pt')
    payload['model_state']['head.1.weight'] = torch.zeros(1)
    save_checkpoint(directory / 'latest.pt', payload)
    monkeypatch.setattr(api, 'make_pretrained_loaders', forbidden)
    with pytest.raises(ValueError, match='tensor|shape'):
        run(setup, resume=True)


def test_each_epoch_requires_full_sample_coverage_before_checkpoint_commit(setup, monkeypatch):
    from torch.utils.data import DataLoader, SequentialSampler
    api, _, _, directory = setup
    def incomplete(*args, **kwargs):
        datasets = {'train': Samples(4794), 'val': Samples(1200)}
        return {name: DataLoader(dataset, batch_size=512, generator=torch.Generator().manual_seed(42),
                                 sampler=SequentialSampler(range(3))) for name, dataset in datasets.items()}
    monkeypatch.setattr(api, 'make_pretrained_loaders', incomplete)
    with pytest.raises(ValueError, match='4794'):
        run(setup)
    assert not (directory / 'latest.pt').exists()
    state = json.loads((directory / 'run_state.json').read_text())
    assert state['status'] == 'failed'
    assert state['epochs_completed'] == 0


def test_json_reports_and_terminal_recovery_survive_moving_run_directory(setup, monkeypatch):
    initial = run(setup)
    api, config, identity, directory = setup
    moved = directory.parent / 'relocated'
    directory.rename(moved)
    monkeypatch.setattr(api, 'make_pretrained_loaders', forbidden)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    result = api.run_training('missing', moved, config, resume=True, device='cpu', tracking='json')
    assert result['history'] == initial['history']
    assert result['checkpoints']['best']['path'] == 'best.pt'
    assert result['checkpoint_reload']['status'] == 'forward_verified'


def test_real_mlflow_interrupted_resume_keeps_run_id_and_metric_steps(setup, monkeypatch):
    pytest.importorskip('mlflow')
    import src.training.trainer as trainer
    from mlflow.tracking import MlflowClient
    api, config, _, directory = setup
    original = trainer.train_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('synthetic interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(trainer, 'train_epoch', interrupt)
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    state = json.loads((directory / 'run_state.json').read_text())
    run_id = state['tracking']['run_id']
    client = MlflowClient(tracking_uri=(directory / 'mlruns').as_uri())
    assert [(m.step, m.value) for m in client.get_metric_history(run_id, 'val_loss')] == [
        (1, read_checkpoint(directory / 'latest.pt')['history'][0]['val_loss'])]
    monkeypatch.setattr(trainer, 'train_epoch', original)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    result = api.run_training('prepared', directory, config, device='cpu', tracking='mlflow', resume=True)
    assert result['tracking']['run_id'] == run_id
    assert [m.step for m in client.get_metric_history(run_id, 'val_loss')] == [1, 2]
    assert client.get_run(run_id).info.status == 'FINISHED'


def test_cli_main_uses_shared_controller_and_terminal_checkpoint_without_pixels(setup, monkeypatch, capsys):
    from src.cli import main
    run(setup)
    api, _, _, directory = setup
    capsys.readouterr()
    monkeypatch.setattr(api, 'validate_preflight_preparation', forbidden)
    monkeypatch.setattr(api, 'make_pretrained_loaders', forbidden)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    exit_code = main(['train', '--task', 'classification', '--config', str(directory / 'config.json'),
                      '--run-dir', str(directory), '--prepared-dir', str(directory.parent / 'missing'),
                      '--device', 'cpu', '--tracking', 'json', '--resume'])
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['optimizer_updates'] == 10
    assert summary['status'] == 'completed'


def test_tracking_failure_marks_mlflow_failed_and_retry_reconciles_same_run(setup, monkeypatch):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    api, config, _, directory = setup
    original = MlflowClient.log_metric
    def unavailable(*args, **kwargs):
        raise RuntimeError('metric storage unavailable')
    monkeypatch.setattr(MlflowClient, 'log_metric', unavailable)
    result = api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    run_id = result['tracking']['run_id']
    client = MlflowClient(tracking_uri=(directory / 'mlruns').as_uri())
    assert client.get_run(run_id).info.status == 'FAILED'
    monkeypatch.setattr(MlflowClient, 'log_metric', original)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    result = api.run_training('missing', directory, config, device='cpu', tracking='mlflow', resume=True)
    assert result['tracking']['run_id'] == run_id
    assert result['tracking']['status'] == 'finished'
    assert [m.step for m in client.get_metric_history(run_id, 'val_loss')] == [1, 2]
    assert any(failure['stage'] == 'tracking' for failure in result['failures'])


@pytest.mark.parametrize('terminal', [False, True])
def test_evidence_failure_records_failed_run_and_terminal_retry_repairs_without_training(setup, monkeypatch, terminal):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    api, config, _, directory = setup
    if terminal:
        api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    original = api._evidence
    failure = OSError('synthetic evidence export failure')
    def unavailable(path):
        raise failure
    monkeypatch.setattr(api, '_evidence', unavailable)
    with pytest.raises(OSError) as caught:
        api.run_training('prepared', directory, config, device='cpu', tracking='mlflow', resume=terminal)
    assert caught.value is failure
    state = json.loads((directory / 'run_state.json').read_text())
    summary = json.loads((directory / 'run_summary.json').read_text())
    assert state['status'] == summary['status'] == 'failed'
    assert state['failures'][-1]['stage'] == 'finalization'
    assert summary['failures'][-1]['message'] == 'synthetic evidence export failure'
    run_id = state['tracking']['run_id']
    client = MlflowClient(tracking_uri=(directory / 'mlruns').as_uri())
    assert client.get_run(run_id).info.status == 'FAILED'
    assert state['tracking']['status'] == 'failed'
    before = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ('best.pt', 'latest.pt')}
    assert read_checkpoint(directory / 'latest.pt')['stop_reason'] == 'completed'
    monkeypatch.setattr(api, '_evidence', original)
    for name in ('fit', 'make_pretrained_loaders', 'build_verified_pretrained_model', 'validate_preflight_preparation'):
        monkeypatch.setattr(api, name, forbidden)
    repaired = api.run_training('missing', directory, config, device='cpu', tracking='mlflow', resume=True)
    assert repaired['status'] == 'completed'
    assert repaired['epochs_completed'] == 2
    assert repaired['optimizer_updates'] == 10
    assert repaired['tracking']['run_id'] == run_id
    assert client.get_run(run_id).info.status == 'FINISHED'
    assert any(item['stage'] == 'finalization' for item in repaired['failures'])
    assert before == {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in before}
    with ZipFile(directory / 'evidence.zip') as archive:
        assert json.loads(archive.read('run_summary.json'))['status'] == 'completed'


def test_best_effort_failure_recording_does_not_mask_original_finalization_error(setup, monkeypatch):
    run(setup)
    api, _, _, _ = setup
    failure = OSError('original export failure')
    def unavailable(*args, **kwargs):
        raise failure
    def cannot_record(*args, **kwargs):
        raise PermissionError('secondary failure recording error')
    monkeypatch.setattr(api, '_evidence', unavailable)
    original_write = api._write_json
    recording_attempted = []
    def write_then_fail_recovery(path, value):
        if isinstance(value, dict) and value.get('status') == 'failed':
            recording_attempted.append(path.name)
            cannot_record()
        return original_write(path, value)
    monkeypatch.setattr(api, '_write_json', write_then_fail_recovery)
    with pytest.raises(OSError) as caught:
        run(setup, resume=True)
    assert caught.value is failure
    assert set(recording_attempted) == {'run_state.json', 'run_summary.json'}


@pytest.mark.parametrize('damage', ['corrupt', 'history', 'tensor', 'provenance'])
def test_nonterminal_resume_rejects_invalid_best_before_pixels_or_updates(setup, monkeypatch, damage):
    import src.training.trainer as trainer
    from src.training.checkpoint import save_checkpoint
    original_train = trainer.train_epoch
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('stop after first checkpoint')
        return original_train(*args, **kwargs)
    monkeypatch.setattr(trainer, 'train_epoch', interrupt)
    with pytest.raises(RuntimeError, match='stop after first checkpoint'):
        run(setup)
    api, _, _, directory = setup
    if damage == 'corrupt':
        (directory / 'best.pt').write_bytes(b'corrupt')
    else:
        best = read_checkpoint(directory / 'best.pt')
        if damage == 'history':
            best['history'][0]['train_loss'] += 1
        elif damage == 'tensor':
            best['model_state']['head.1.weight'] = torch.zeros(1)
        else:
            best['weight_provenance'] = {'status': 'conflicting initial weights'}
        save_checkpoint(directory / 'best.pt', best)
    before = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ('best.pt', 'latest.pt')}
    monkeypatch.setattr(api, 'make_pretrained_loaders', forbidden)
    monkeypatch.setattr(api, 'fit', forbidden)
    monkeypatch.setattr(trainer, 'train_epoch', forbidden)
    with pytest.raises(ValueError):
        run(setup, resume=True)
    assert before == {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in before}
    assert read_checkpoint(directory / 'latest.pt')['epochs_completed'] == 1


@pytest.mark.parametrize('contents', [b'{malformed', b'\xff', b'[]', b'{"checkpoints":[]}'])
def test_terminal_resume_ignores_unreadable_optional_summary_and_records_no_forward(setup, monkeypatch, contents):
    initial = run(setup)
    api, _, _, directory = setup
    (directory / 'run_summary.json').write_bytes(contents)
    for name in ('fit', 'make_pretrained_loaders', 'build_verified_pretrained_model', 'validate_preflight_preparation'):
        monkeypatch.setattr(api, name, forbidden)
    result = run(setup, resume=True)
    assert result['history'] == initial['history']
    assert result['checkpoint_reload']['status'] == 'state_loaded_without_forward'
    assert json.loads((directory / 'run_summary.json').read_text())['status'] == 'completed'


def test_secondary_report_failure_preserves_original_training_error_and_checkpoint(setup, monkeypatch):
    import src.training.trainer as trainer
    api, _, _, directory = setup
    original_train = trainer.train_epoch
    original_write = api._write_json
    failure = RuntimeError('original training failure')
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original_train(*args, **kwargs)
    def unavailable_failure_report(path, value):
        if isinstance(value, dict) and value.get('status') == 'failed':
            raise PermissionError('secondary report failure')
        return original_write(path, value)
    monkeypatch.setattr(trainer, 'train_epoch', interrupted)
    monkeypatch.setattr(api, '_write_json', unavailable_failure_report)
    with pytest.raises(RuntimeError) as caught:
        run(setup)
    assert caught.value is failure
    assert read_checkpoint(directory / 'latest.pt')['epochs_completed'] == 1


def test_completion_log_append_failure_records_failed_and_terminal_resume_repairs(setup, monkeypatch):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    api, config, _, directory = setup
    original_open = Path.open
    failure = OSError('completion log append unavailable')
    checkpoints = {}
    def fail_completion_append(path, mode='r', *args, **kwargs):
        if path == directory / 'run.log' and mode == 'a' and (directory / 'latest.pt').exists():
            if read_checkpoint(directory / 'latest.pt')['stop_reason'] == 'completed':
                checkpoints.update({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                    for name in ('best.pt', 'latest.pt')})
                raise failure
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_completion_append)
    with pytest.raises(OSError) as caught:
        api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    assert caught.value is failure
    state = json.loads((directory / 'run_state.json').read_text())
    summary = json.loads((directory / 'run_summary.json').read_text())
    assert state['status'] == summary['status'] == 'failed'
    assert state['failures'][-1]['stage'] == 'finalization'
    assert state['failures'][-1]['message'] == 'completion log append unavailable'
    run_id = state['tracking']['run_id']
    client = MlflowClient(tracking_uri=(directory / 'mlruns').as_uri())
    assert client.get_run(run_id).info.status == 'FAILED'
    assert checkpoints == {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in checkpoints}
    monkeypatch.setattr(Path, 'open', original_open)
    for name in ('fit', 'make_pretrained_loaders', 'build_verified_pretrained_model', 'validate_preflight_preparation'):
        monkeypatch.setattr(api, name, forbidden)
    repaired = api.run_training('missing', directory, config, device='cpu', tracking='mlflow', resume=True)
    assert repaired['status'] == 'completed'
    assert repaired['epochs_completed'] == 2
    assert repaired['optimizer_updates'] == 10
    assert repaired['tracking']['run_id'] == run_id
    assert client.get_run(run_id).info.status == 'FINISHED'
    assert checkpoints == {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in checkpoints}
    assert any(item['stage'] == 'finalization' for item in repaired['failures'])


def test_initial_checkpoint_reuses_weights_with_fresh_training_state_and_provenance(setup, monkeypatch):
    import src.training.trainer as trainer
    import src.training.checkpoint as checkpoint
    run(setup)
    api, config, identity, parent_dir = setup
    parent = parent_dir / 'best.pt'
    # Preserve an actual legacy on-disk config without the newly optional keys.
    legacy = torch.load(parent, weights_only=True, map_location='cpu')
    legacy['training_config'].pop('selection_metric', None)
    for phase in legacy['training_config']['phases']:
        phase.pop('backbone_lr', None)
    torch.save(legacy, parent)
    parent.with_suffix('.pt.sha256').write_text(hashlib.sha256(parent.read_bytes()).hexdigest() + '\n')
    saved = read_checkpoint(parent)
    assert saved['model_config']['fine_tuning'] == 'frozen'
    before = {path: path.read_bytes() for path in (parent, parent.with_suffix('.pt.sha256'))}
    new_config = replace(config, phases=(Phase('partial', 1, 0.003, backbone_lr=0.00003),),
                         selection_metric='val_macro_f1')
    original_epoch = trainer.train_epoch
    observed = []

    def inspect_epoch(model, loader, optimizer, *args, **kwargs):
        assert model.config.fine_tuning == 'partial'
        assert all(parameter.requires_grad for parameter in model.encoder.parameters())
        assert not optimizer.state
        assert [group['lr'] for group in optimizer.param_groups] == [0.003, 0.00003]
        assert {id(p) for p in optimizer.param_groups[0]['params']} == {id(p) for p in model.head.parameters()}
        assert {id(p) for p in optimizer.param_groups[1]['params']} == {id(p) for p in model.encoder.parameters()}
        assert torch.equal(torch.get_rng_state(), torch.Generator().manual_seed(42).get_state())
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, saved['model_state'][name], rtol=0, atol=0)
        observed.append(True)
        return original_epoch(model, loader, optimizer, *args, **kwargs)

    monkeypatch.setattr(trainer, 'train_epoch', inspect_epoch)
    monkeypatch.setattr(api, 'build_verified_pretrained_model', forbidden)
    monkeypatch.setattr(checkpoint, 'restore_rng_state', forbidden)
    result = api.run_training(parent_dir.parent / 'prepared', parent_dir.parent / 'warmstart', new_config,
                              initial_checkpoint=parent, device='cpu', tracking='json')
    assert observed == [True]
    assert result['epochs_completed'] == 1
    assert result['optimizer_updates'] == 5
    assert result['history'][0]['epoch'] == 1
    provenance = result['weight_provenance']
    assert provenance['sha256'] == 'c' * 64
    initial = provenance['initial_checkpoint']
    assert initial['filename'] == 'best.pt'
    assert initial['sha256'] == hashlib.sha256(before[parent]).hexdigest()
    assert initial['epochs_completed'] == saved['epochs_completed']
    assert initial['weight_provenance'] == saved['weight_provenance']
    child = read_checkpoint(parent_dir.parent / 'warmstart' / 'latest.pt')
    assert child['weight_provenance'] == provenance
    assert child['training_config'] == new_config.to_dict()
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize('damage', ['identity', 'architecture', 'head_dropout', 'freeze_bn_stats', 'tensor', 'corrupt'])
def test_initial_checkpoint_rejects_incompatible_parent_before_pixels_or_updates(setup, monkeypatch, damage):
    from src.training.checkpoint import save_checkpoint
    run(setup)
    api, config, identity, parent_dir = setup
    parent = parent_dir / 'best.pt'
    if damage == 'identity':
        monkeypatch.setattr(api, 'validate_preflight_preparation', lambda path: dict(identity, class_names=list(reversed(identity['class_names']))))
    elif damage in ('architecture', 'head_dropout', 'freeze_bn_stats'):
        values = {'architecture': 'efficientnet_b3', 'head_dropout': 0.4, 'freeze_bn_stats': False}
        config = replace(config, **{damage: values[damage]})
    elif damage == 'tensor':
        payload = read_checkpoint(parent)
        payload['model_state']['head.1.weight'] = torch.zeros(1)
        save_checkpoint(parent, payload)
    else:
        parent.write_bytes(b'corrupt parent')
    before = {path: path.read_bytes() for path in (parent, parent.with_suffix('.pt.sha256'))}
    for name in ('make_pretrained_loaders', 'build_verified_pretrained_model', 'fit'):
        monkeypatch.setattr(api, name, forbidden)
    with pytest.raises(ValueError):
        api.run_training(parent_dir.parent / 'prepared', parent_dir.parent / 'warmstart', config,
                         initial_checkpoint=parent, device='cpu', tracking='json')
    assert all(path.read_bytes() == contents for path, contents in before.items())


def test_initial_checkpoint_and_resume_are_mutually_exclusive_before_io(setup, monkeypatch):
    api, config, _, directory = setup
    for name in ('read_checkpoint', 'validate_preflight_preparation', 'make_pretrained_loaders'):
        monkeypatch.setattr(api, name, forbidden)
    with pytest.raises(ValueError, match='initial_checkpoint.*resume|resume.*initial_checkpoint'):
        run(setup, resume=True, initial_checkpoint=directory / 'missing.pt')


def test_optional_backbone_lr_exports_and_tracks_across_frozen_and_legacy_rows(setup, monkeypatch):
    pytest.importorskip('mlflow')
    from mlflow.tracking import MlflowClient
    from src.training.checkpoint import save_checkpoint
    api, config, _, directory = setup
    config = replace(config, phases=(Phase('frozen', 1, 0.001),
                                     Phase('partial', 1, 0.001, backbone_lr=0.00003)))
    result = api.run_training('prepared', directory, config, device='cpu', tracking='mlflow')
    client = MlflowClient(tracking_uri=(directory / 'mlruns').as_uri())
    run_id = result['tracking']['run_id']
    assert [(point.step, point.value) for point in client.get_metric_history(run_id, 'backbone_learning_rate')] == [(2, 0.00003)]
    for name in ('best.pt', 'latest.pt'):
        payload = read_checkpoint(directory / name)
        payload['history'][0].pop('backbone_learning_rate')
        save_checkpoint(directory / name, payload)
    for name in ('fit', 'make_pretrained_loaders', 'build_verified_pretrained_model', 'validate_preflight_preparation'):
        monkeypatch.setattr(api, name, forbidden)
    repaired = api.run_training('missing', directory, config, device='cpu', tracking='mlflow', resume=True)
    assert repaired['status'] == 'completed'
    assert repaired['tracking']['status'] == 'finished'
    with (directory / 'history.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['backbone_learning_rate'] == ''
    assert float(rows[1]['backbone_learning_rate']) == 0.00003
    assert [(point.step, point.value) for point in client.get_metric_history(run_id, 'backbone_learning_rate')] == [(2, 0.00003)]
