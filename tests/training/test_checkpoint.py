"""Checkpoint integrity and exact RNG recovery, using synthetic CPU evidence only."""
import copy
from dataclasses import asdict
import hashlib
import importlib
import random

import numpy as np
import pytest
import torch

from test_trainer import DATA_IDENTITY, PROVENANCE, TinyAdapter, tiny_loaders


@pytest.fixture
def api():
    assert importlib.util.find_spec('src.training.checkpoint') is not None, 'Checkpoint helpers are missing'
    return importlib.import_module('src.training.checkpoint')


@pytest.fixture
def payload(api, tmp_path):
    from src.training.trainer import TrainConfig, Phase, fit
    config = TrainConfig(amp='off', batch_size=3, head_dropout=0.25, phases=(Phase(epochs=1),))
    fit(TinyAdapter(), tiny_loaders(), config, tmp_path / 'source',
        data_identity=DATA_IDENTITY, weight_provenance=PROVENANCE)
    return api.read_checkpoint(tmp_path / 'source' / 'latest.pt')


def test_safe_roundtrip_and_expected_metadata(api, payload, tmp_path):
    path = tmp_path / 'copy.pt'
    api.save_checkpoint(path, payload)
    loaded = api.read_checkpoint(path, expected_config=payload['training_config'],
                                 expected_data_identity=DATA_IDENTITY)
    assert loaded['history'] == payload['history']
    assert loaded['weight_provenance'] == PROVENANCE
    assert loaded['class_names'] == DATA_IDENTITY['class_names']
    assert torch.equal(loaded['model_state']['head.1.weight'], payload['model_state']['head.1.weight'])
    safe = torch.load(path, weights_only=True, map_location='cpu')
    assert safe['epochs_completed'] == 1
    assert path.with_suffix('.pt.sha256').read_text().strip() == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('damage', ['missing_hash', 'wrong_hash', 'truncated', 'corrupt'])
def test_read_fails_closed_on_integrity_damage(api, payload, tmp_path, damage):
    path = tmp_path / 'copy.pt'
    api.save_checkpoint(path, payload)
    sidecar = path.with_suffix('.pt.sha256')
    if damage == 'missing_hash':
        sidecar.unlink()
    elif damage == 'wrong_hash':
        sidecar.write_text('0' * 64)
    else:
        path.write_bytes(path.read_bytes()[:40] if damage == 'truncated' else b'not a checkpoint')
    with pytest.raises(ValueError, match='SHA256|integrity'):
        api.read_checkpoint(path)


def test_corrupt_serialization_with_matching_hash_rejected(api, tmp_path):
    path = tmp_path / 'broken.pt'
    path.write_bytes(b'not a checkpoint')
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='checkpoint'):
        api.read_checkpoint(path)


@pytest.mark.parametrize('change', [
    {'schema_version': 2}, {'epochs_completed': 0}, {'phase_index': 20}, {'epoch_in_phase': 0},
    {'optimizer_updates': 0}, {'class_names': ['wrong'] * 200}, {'resolved_precision': 'auto'},
    {'history': []}, {'stop_reason': 'anything'}, {'model_config': {}},
])
def test_read_rejects_invalid_schema_or_progress(api, payload, tmp_path, change):
    damaged = {**payload, **change}
    path = tmp_path / 'invalid.pt'
    # Bypass the writer to check untrusted on-disk validation itself.
    torch.save(damaged, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        api.read_checkpoint(path)


def test_missing_required_state_rejected(api, payload, tmp_path):
    del payload['optimizer_state']
    path = tmp_path / 'invalid.pt'
    torch.save(payload, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='missing|schema'):
        api.read_checkpoint(path)


def test_expected_config_data_and_class_mapping_rejected(api, payload, tmp_path):
    path = tmp_path / 'copy.pt'
    api.save_checkpoint(path, payload)
    with pytest.raises(ValueError, match='config'):
        api.read_checkpoint(path, expected_config={**payload['training_config'], 'seed': 7})
    changed = {**DATA_IDENTITY, 'class_names': list(reversed(DATA_IDENTITY['class_names']))}
    with pytest.raises(ValueError, match='data'):
        api.read_checkpoint(path, expected_data_identity=changed)


def test_rng_roundtrip_covers_python_numpy_torch_and_both_loader_generators(api):
    loaders = tiny_loaders()
    state = api.capture_rng_state(loaders)
    def draw():
        return (random.random(), np.random.random(), torch.rand(4),
                {name: torch.rand(4, generator=loader.generator) for name, loader in loaders.items()})
    expected = draw()
    draw()
    api.restore_rng_state(state, loaders)
    actual = draw()
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    for name in loaders:
        assert torch.equal(actual[3][name], expected[3][name])


def test_failed_temp_write_preserves_previous_checkpoint(api, payload, tmp_path, monkeypatch):
    path = tmp_path / 'copy.pt'
    api.save_checkpoint(path, payload)
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('simulated disk failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='disk failure'):
        api.save_checkpoint(path, payload)
    assert path.read_bytes() == original
    assert api.read_checkpoint(path)['history'] == payload['history']
    assert sorted(item.name for item in tmp_path.glob('copy*')) == ['copy.pt', 'copy.pt.sha256']


def test_public_model_reload_is_offline_strict_and_portable_to_cpu(api, payload, tmp_path, monkeypatch):
    from src.models.pretrained import ModelConfig, create_pretrained_model
    from src.training.trainer import TrainConfig, Phase
    def no_download(*args, **kwargs):
        raise AssertionError('checkpoint loading must never fetch ImageNet weights')
    monkeypatch.setattr(torch.hub, 'download_url_to_file', no_download)
    model = create_pretrained_model(ModelConfig('efficientnet_b3', fine_tuning='partial', head_dropout=0.25),
                                    load_weights=False)
    payload['model_config'] = asdict(model.config)
    payload['model_state'] = copy.deepcopy(model.state_dict())
    payload['training_config'] = TrainConfig(architecture='efficientnet_b3', amp='float16',
        phases=(Phase('partial', 1),), head_dropout=0.25).to_dict()
    payload['resolved_precision'] = 'float16'
    payload['history'][0]['phase_mode'] = 'partial'
    payload['scaler_state'] = {'scale': 128.0, 'growth_factor': 2.0, 'backoff_factor': 0.5,
                              'growth_interval': 2000, '_growth_tracker': 0}
    path = tmp_path / 'real-adapter.pt'
    api.save_checkpoint(path, payload)
    loaded, restored = api.load_model_checkpoint(path, expected_data_identity=DATA_IDENTITY)
    assert loaded.config == model.config
    assert not loaded.training
    assert next(loaded.parameters()).device.type == 'cpu'
    assert {name: p.requires_grad for name, p in loaded.named_parameters()} == {
        name: p.requires_grad for name, p in model.named_parameters()}
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert restored['resolved_precision'] == 'float16'
    model.eval()
    with torch.inference_mode():
        images = torch.ones(1, 3, 32, 32)
        assert torch.equal(model(images), loaded(images))
    # Strict loading must reject dtype coercion as well as key/shape mismatches.
    for damage in ('dtype', 'shape', 'missing', 'extra'):
        broken = dict(payload['model_state'])
        if damage == 'dtype':
            broken['head.1.weight'] = broken['head.1.weight'].double()
        elif damage == 'shape':
            broken['head.1.weight'] = broken['head.1.weight'][:1]
        elif damage == 'missing':
            broken.pop('head.1.weight')
        else:
            broken['unexpected'] = torch.ones(1)
        api.save_checkpoint(path, {**payload, 'model_state': broken})
        with pytest.raises(ValueError, match='state|tensor'):
            api.load_model_checkpoint(path)

@pytest.mark.parametrize('change', [
    {'data_identity': {'class_names': DATA_IDENTITY['class_names']}},
    {'optimizer_state': {}}, {'scheduler_state': {}}, {'validation': {'loss': 5}},
    {'rng_state': {}},
])
def test_read_rejects_missing_recovery_metadata(api, payload, tmp_path, change):
    path = tmp_path / 'invalid-recovery.pt'
    torch.save({**payload, **change}, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        api.read_checkpoint(path)


def test_read_never_falls_back_to_unsafe_pickle(api, payload, tmp_path):
    path = tmp_path / 'unsafe.pt'
    torch.save({**payload, 'custom_object': TinyAdapter()}, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='safe checkpoint'):
        api.read_checkpoint(path)


def test_read_requires_complete_validation_report_for_recovery(api, payload, tmp_path):
    payload['validation'] = {'loss': payload['validation']['loss']}
    path = tmp_path / 'invalid-report.pt'
    torch.save(payload, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='validation'):
        api.read_checkpoint(path)


@pytest.mark.parametrize('mode', ['frozen', 'partial'])
def test_schema_one_without_new_config_fields_resumes_legacy_optimizer(api, tmp_path, mode, monkeypatch):
    from src.training.trainer import TrainConfig, Phase, fit
    config = TrainConfig(amp='off', batch_size=3, head_dropout=0.25,
                         phases=(Phase(mode, 2, 0.01),), lr_decay=0.5)
    def interrupt(record, validation):
        if record['epoch'] == 2:
            raise RuntimeError('interrupted')
    torch.manual_seed(19)
    fit(TinyAdapter(), tiny_loaders(), config, tmp_path / 'full',
        data_identity=DATA_IDENTITY, weight_provenance=PROVENANCE)
    torch.manual_seed(19)
    with pytest.raises(RuntimeError, match='interrupted'):
        fit(TinyAdapter(), tiny_loaders(), config, tmp_path / 'legacy', on_epoch=interrupt,
            data_identity=DATA_IDENTITY, weight_provenance=PROVENANCE)
    path = tmp_path / 'legacy' / 'latest.pt'
    legacy = api.read_checkpoint(path)
    assert len(legacy['optimizer_state']['param_groups']) == 1
    assert legacy['training_config'].pop('selection_metric') == 'val_loss'
    assert legacy['training_config'].pop('label_smoothing') == 0.0
    assert legacy['training_config'].pop('color_jitter') == 0.0
    assert legacy['training_config'].pop('mixup_alpha') == 0.0
    assert legacy['training_config'].pop('weights') == 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1'
    assert legacy['training_config']['phases'][0].pop('backbone_lr') is None
    for row in legacy['history']:
        row.pop('backbone_learning_rate')
    torch.save(legacy, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = api.read_checkpoint(path, expected_config=config)
    assert loaded['training_config'] == config.to_dict()
    raw_bytes = path.read_bytes()
    def offline_adapter(model_config, *, load_weights):
        assert load_weights is False
        model = TinyAdapter(dropout=model_config.head_dropout)
        model.set_fine_tuning(model_config.fine_tuning, freeze_bn_stats=model_config.freeze_bn_stats)
        return model
    monkeypatch.setattr(api, 'create_pretrained_model', offline_adapter)
    model, restored = api.load_model_checkpoint(path, expected_data_identity=DATA_IDENTITY)
    assert restored['training_config']['mixup_alpha'] == 0.0 and not model.training
    assert all(torch.equal(value, legacy['model_state'][name]) for name, value in model.state_dict().items())
    assert path.read_bytes() == raw_bytes
    resumed = fit(TinyAdapter(), tiny_loaders(), config, path.parent, resume=True,
                  data_identity=DATA_IDENTITY, weight_provenance=PROVENANCE)
    expected = api.read_checkpoint(tmp_path / 'full' / 'latest.pt')
    assert resumed['best_epoch'] == expected['best_epoch']
    assert resumed['history'][-1] == expected['history'][-1]
    actual = api.read_checkpoint(path)
    for name, tensor in expected['model_state'].items():
        assert torch.equal(tensor, actual['model_state'][name]), name


def test_checkpoint_rejects_later_tie_as_selected_best(api, payload, tmp_path):
    payload['training_config']['phases'][0]['epochs'] = 2
    payload['epochs_completed'] = payload['epoch_in_phase'] = 2
    payload['optimizer_updates'] *= 2
    payload['history'].append({**payload['history'][0], 'epoch': 2, 'epoch_in_phase': 2,
                              'total_optimizer_updates': payload['optimizer_updates']})
    payload['best_epoch'] = 2
    with pytest.raises(ValueError, match='best|validation'):
        api.save_checkpoint(tmp_path / 'late-tie.pt', payload)


def test_checkpoint_rejects_mismatched_model_and_training_weight_profiles(api, payload, tmp_path):
    from src.models.pretrained import FB22K_WEIGHT_ID
    payload['training_config']['weights'] = FB22K_WEIGHT_ID
    with pytest.raises(ValueError, match='model config'):
        api.save_checkpoint(tmp_path / 'profile-mismatch.pt', payload)
