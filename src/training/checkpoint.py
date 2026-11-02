"""Portable weights-only checkpoints and completed-epoch random-state recovery."""
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np
import torch

from src.models.pretrained import ModelConfig, create_pretrained_model


_REQUIRED = {
    'schema_version', 'model_state', 'model_config', 'training_config', 'weight_provenance',
    'data_identity', 'class_names', 'optimizer_state', 'scheduler_state', 'scaler_state',
    'epochs_completed', 'phase_index', 'epoch_in_phase', 'resolved_precision', 'optimizer_updates',
    'history', 'best_loss', 'best_epoch', 'early_stop_best', 'bad_epochs', 'stop_reason',
    'validation', 'rng_state',
}


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _validate(payload):
    """Check the small public format before any restoration or model construction."""
    from src.training.trainer import TrainConfig
    if not isinstance(payload, dict) or _REQUIRED - payload.keys():
        raise ValueError('Checkpoint schema has missing required fields')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 1:
        raise ValueError('Unsupported checkpoint schema version')
    try:
        config = TrainConfig.from_dict(payload['training_config'])
        model_config = ModelConfig(**payload['model_config'])
        manifest = payload['data_identity'].get('manifest_sha256')
        if not isinstance(manifest, str) or len(manifest) != 64 or any(c not in '0123456789abcdef' for c in manifest):
            raise ValueError('Checkpoint data identity requires manifest_sha256')
        names = payload['class_names']
        if (not isinstance(names, list) or len(names) != 200
                or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 200
                or names != list(payload['data_identity']['class_names'])):
            raise ValueError('Checkpoint class mapping mismatch')
        for key, minimum in (('epochs_completed', 1), ('phase_index', 0), ('epoch_in_phase', 1),
                             ('optimizer_updates', 1), ('best_epoch', 1), ('bad_epochs', 0)):
            if type(payload[key]) is not int or payload[key] < minimum:
                raise ValueError(f'Invalid checkpoint progress: {key}')
        phase_index = payload['phase_index']
        if phase_index >= len(config.phases):
            raise ValueError('Invalid checkpoint phase_index')
        phase = config.phases[phase_index]
        completed = payload['epochs_completed']
        if (payload['epoch_in_phase'] > phase.epochs
                or completed != sum(p.epochs for p in config.phases[:phase_index]) + payload['epoch_in_phase']):
            raise ValueError('Inconsistent checkpoint epoch progress')
        if (model_config.architecture != config.architecture or model_config.weights != config.weights
                or model_config.fine_tuning != phase.mode
                or model_config.head_dropout != config.head_dropout or model_config.freeze_bn_stats != config.freeze_bn_stats):
            raise ValueError('Checkpoint model config differs from training config')
        precision = payload['resolved_precision']
        if (precision not in ('fp32', 'float16', 'bfloat16')
                or config.amp != 'auto' and precision != {'off': 'fp32', 'float16': 'float16', 'bfloat16': 'bfloat16'}[config.amp]
                or (precision == 'float16') != isinstance(payload['scaler_state'], dict)):
            raise ValueError('Invalid checkpoint resolved precision or scaler state')
        reason = payload['stop_reason']
        total_epochs = sum(p.epochs for p in config.phases)
        early_stopped = config.patience is not None and payload['bad_epochs'] >= config.patience
        if (reason not in (None, 'completed', 'early_stopping')
                or (reason == 'early_stopping') != early_stopped
                or reason == 'completed' and completed != total_epochs
                or reason is None and completed == total_epochs):
            raise ValueError('Inconsistent checkpoint stop_reason')
        history = payload['history']
        if (not isinstance(history, list) or len(history) != completed
                or [row['epoch'] for row in history] != list(range(1, completed + 1))
                or history[-1]['phase_index'] != phase_index
                or history[-1]['epoch_in_phase'] != payload['epoch_in_phase']
                or sum(row['optimizer_updates'] for row in history) != payload['optimizer_updates']
                or any(row['optimizer_updates'] < 1 for row in history)):
            raise ValueError('Inconsistent checkpoint history')
        selected = [row[config.selection_metric] for row in history]
        optimum = max(selected) if config.selection_metric == 'val_macro_f1' else min(selected)
        if (payload['best_epoch'] > completed
                or any(type(payload[key]) not in (int, float) or not math.isfinite(payload[key])
                       for key in ('best_loss', 'early_stop_best'))
                or payload['best_loss'] != min(row['val_loss'] for row in history)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in selected)
                or payload['best_epoch'] != selected.index(optimum) + 1
                or payload['validation']['loss'] != history[-1]['val_loss']):
            raise ValueError('Inconsistent checkpoint validation or best loss')
        for key in ('model_state', 'weight_provenance', 'data_identity', 'optimizer_state',
                    'scheduler_state', 'validation', 'rng_state'):
            if not isinstance(payload[key], dict):
                raise ValueError(f'Invalid checkpoint {key}')
        if not {'state', 'param_groups'} <= payload['optimizer_state'].keys():
            raise ValueError('Incomplete checkpoint optimizer state')
        if not {'step_size', 'gamma', 'last_epoch', 'base_lrs', '_last_lr'} <= payload['scheduler_state'].keys():
            raise ValueError('Incomplete checkpoint scheduler state')
        if not {'loss', 'top1_accuracy', 'top5_accuracy', 'macro_precision', 'macro_recall',
                'macro_f1', 'num_samples', 'per_class', 'confusion_matrix', 'predictions'} <= payload['validation'].keys():
            raise ValueError('Incomplete checkpoint validation report')
        if set(payload['rng_state']) != {'python', 'numpy', 'torch_cpu', 'torch_cuda', 'loader_generators'}:
            raise ValueError('Invalid checkpoint RNG state')
        payload['training_config'] = config.to_dict()
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('Invalid checkpoint schema, config or progress') from error
    return payload


def save_checkpoint(path, payload):
    """Replace each file atomically; a two-file interruption fails closed on read.

    No journal or unsafe pickle fallback. The digest is an integrity check, not
    authentication. Best and latest must each be saved from a completed epoch.
    """
    _validate(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = digest_temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = _sha256(temporary)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp',
                                         mode='w', encoding='ascii', delete=False) as stream:
            digest_temporary = Path(stream.name)
            stream.write(digest + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.replace(digest_temporary, path.with_suffix(path.suffix + '.sha256'))
    finally:
        for leftover in (temporary, digest_temporary):
            if leftover is not None:
                leftover.unlink(missing_ok=True)


def read_checkpoint(path, *, expected_config=None, expected_data_identity=None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_suffix(path.suffix + '.sha256')
    try:
        digest = sidecar.read_text(encoding='ascii').strip()
    except (OSError, UnicodeError) as error:
        raise ValueError('Checkpoint SHA256 sidecar missing or unreadable') from error
    if digest != _sha256(path):
        raise ValueError('Checkpoint SHA256 integrity mismatch')
    try:
        payload = torch.load(path, weights_only=True, map_location='cpu')
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError, IndexError) as error:
        raise ValueError('Cannot read safe checkpoint payload') from error
    payload = _validate(payload)
    if expected_config is not None:
        from src.training.trainer import TrainConfig
        expected = expected_config.to_dict() if isinstance(expected_config, TrainConfig) else TrainConfig.from_dict(expected_config).to_dict()
        if payload['training_config'] != expected:
            raise ValueError('Checkpoint training config mismatch')
    if expected_data_identity is not None and _json_value(payload['data_identity']) != _json_value(expected_data_identity):
        raise ValueError('Checkpoint data identity mismatch')
    return payload


def capture_rng_state(loaders):
    """Encode NumPy state as tensors/primitives accepted by weights_only=True."""
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (algorithm, torch.tensor(keys.astype(np.int64)), position, has_gauss, cached_gauss),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader_generators': {name: loader.generator.get_state() for name, loader in loaders.items()}}


def restore_rng_state(state, loaders):
    """Restore last, after model/optimizer construction and state loading."""
    if set(state['loader_generators']) != set(loaders):
        raise ValueError('Checkpoint loader generator names differ')
    cuda_states = state['torch_cuda']
    if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
        raise ValueError('Checkpoint CUDA RNG device count differs')
    random.setstate(state['python'])
    algorithm, keys, position, has_gauss, cached_gauss = state['numpy']
    np.random.set_state((algorithm, keys.numpy().astype(np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state['torch_cpu'])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    for name, loader in loaders.items():
        loader.generator.set_state(state['loader_generators'][name])


def _load_model_state(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Checkpoint model state keys differ')
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f'Checkpoint model tensor shape/dtype differs: {name}')
    model.load_state_dict(state, strict=True)


def load_model_checkpoint(path, *, device='cpu', expected_data_identity=None):
    """Reload the saved adapter and policy offline, independent of training AMP."""
    payload = read_checkpoint(path, expected_data_identity=expected_data_identity)
    model = create_pretrained_model(ModelConfig(**payload['model_config']), load_weights=False)
    _load_model_state(model, payload['model_state'])
    model.to(device).eval()
    return model, payload
