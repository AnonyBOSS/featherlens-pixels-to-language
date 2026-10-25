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
