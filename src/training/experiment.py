"""One classification experiment with portable reports and optional local MLflow.

Use one writer per run directory. latest.pt owns completed-epoch recovery; JSON
and CSV are rebuilt from checkpoints. Evidence ZIPs omit pixels and weights.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from src.models.pretrained import ModelConfig
from src.training.checkpoint import load_model_checkpoint, read_checkpoint
from src.training.data import PreprocessingConfig, make_pretrained_loaders
from src.training.runtime import collect_runtime, validate_preflight_preparation, _failure
from src.training.trainer import TrainConfig, fit, resolve_precision
from src.training.weights import build_verified_pretrained_model


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _log(directory, message):
    print(message, flush=True)
    with (directory / 'run.log').open('a', encoding='utf-8') as stream:
        stream.write(message + '\n')


def _tracking_error(directory, state, error, client=None):
    failure = _failure('tracking', error)
    state['failures'].append(failure)
    state['tracking']['status'] = 'error'
    print(f"Tracking error: {failure['type']}: {failure['message']}", file=sys.stderr, flush=True)
    with (directory / 'run.log').open('a', encoding='utf-8') as stream:
        stream.write(f"Tracking error: {failure['message']}\n")
    if client is not None:
        try:
            client.set_terminated(state['tracking']['run_id'], status='FAILED')
        except Exception as termination_error:
            state['failures'].append(_failure('tracking', termination_error))


def _open_tracking(directory, state):
    """Use only the local client; no fluent active run, server or autologging."""
    if state['tracking']['mode'] == 'json':
        return None
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as error:
        raise ImportError('Requested MLflow is missing; install mlflow-skinny==3.17.2 or use --tracking json') from error
    try:
        client = MlflowClient(tracking_uri=(directory / 'mlruns').as_uri())
        if state['tracking'].get('run_id'):
            client.get_run(state['tracking']['run_id'])
            client.update_run(state['tracking']['run_id'], status='RUNNING')
        else:
            experiment = client.get_experiment_by_name('FeatherLens-classification')
            experiment_id = (experiment.experiment_id if experiment else
                             client.create_experiment('FeatherLens-classification'))
            state['tracking']['run_id'] = client.create_run(experiment_id).info.run_id
        state['tracking']['status'] = 'running'
        state['tracking']['version'] = mlflow.__version__
        return client
    except Exception as error:
        _tracking_error(directory, state, error)
        return None


def _sync_tracking(client, directory, state, config, history):
    if client is None:
        return
    try:
        run_id = state['tracking']['run_id']
        tracked = client.get_run(run_id)
        params = {'architecture': config.architecture, 'seed': config.seed, 'batch_size': config.batch_size,
                  'accumulation_steps': config.accumulation_steps, 'amp': config.amp,
                  'phases': json.dumps(config.to_dict()['phases'], separators=(',', ':')),
                  'manifest_sha256': state['data_identity']['manifest_sha256']}
        for key, value in params.items():
            if key not in tracked.data.params:
                client.log_param(run_id, key, value)
            elif tracked.data.params[key] != str(value):
                raise ValueError(f'MLflow parameter differs from saved run: {key}')
        # A restart may follow a logging interruption. Fill missing points only.
        metric_names = [key for key, value in history[0].items()
                        if type(value) in (int, float) and key != 'epoch'] if history else []
        for key in metric_names:
            logged = {point.step: point.value for point in client.get_metric_history(run_id, key)}
            for row in history:
                if row['epoch'] in logged:
                    if logged[row['epoch']] != row[key]:
                        raise ValueError(f'MLflow metric differs from checkpoint: {key}')
                else:
                    client.log_metric(run_id, key, row[key], step=row['epoch'])
        for filename in ('config.json', 'data_identity.json', 'history.csv',
                         'validation-best.json', 'validation-latest.json'):
            if (directory / filename).is_file():
                client.log_artifact(run_id, str(directory / filename))
        client.set_terminated(run_id, status='FINISHED' if state['status'] == 'completed' else 'FAILED')
        state['tracking']['status'] = 'finished' if state['status'] == 'completed' else 'failed'
    except Exception as error:
        _tracking_error(directory, state, error, client)


def _check_counts(record):
    if record['num_train_samples'] != 4794 or record['num_val_samples'] != 1200:
        raise ValueError('Each completed epoch must cover 4794 train and 1200 validation samples')


def _validated_best(directory, config, latest):
    best = read_checkpoint(directory / 'best.pt', expected_config=config,
                           expected_data_identity=latest['data_identity'])
    if (best['epochs_completed'] != latest['best_epoch']
            or best['history'] != latest['history'][:best['epochs_completed']]
            or best['weight_provenance'] != latest['weight_provenance']
            or best['resolved_precision'] != latest['resolved_precision']):
        raise ValueError('Best checkpoint does not agree with authoritative latest checkpoint')
    for record in latest['history']:
        _check_counts(record)
    return best


def _export_checkpoints(directory, config, identity):
    latest = read_checkpoint(directory / 'latest.pt', expected_config=config, expected_data_identity=identity)
    best = _validated_best(directory, config, latest)
    _write_json(directory / 'history.json', latest['history'])
    _write_csv(directory / 'history.csv', latest['history'], list(latest['history'][0]))
    for suffix, payload in (('best', best), ('latest', latest)):
        validation = payload['validation']
        _write_json(directory / f'validation-{suffix}.json',
                    {key: value for key, value in validation.items() if key != 'predictions'})
        rows = [dict(prediction, image_id=image_id, top5=json.dumps(prediction['top5']))
                for image_id, prediction in validation['predictions'].items()]
        _write_csv(directory / f'predictions-{suffix}.csv', rows,
                   ['image_id', 'true_label', 'predicted_label', 'top5', 'confidence'])
    references = {name: {'path': f'{name}.pt', 'sha256': _sha256(directory / f'{name}.pt')}
                  for name in ('best', 'latest')}
    return latest, best, references


def _evidence(directory):
    # An explicit allowlist avoids accidental dataset/model/MLflow inclusion.
    files = ['config.json', 'data_identity.json', 'run_state.json', 'history.json', 'history.csv',
             'validation-best.json', 'validation-latest.json', 'predictions-best.csv',
             'predictions-latest.csv', 'run_summary.json', 'run.log']
    files = [filename for filename in files if (directory / filename).is_file()]
    _write_json(directory / 'sha256.json', {filename: _sha256(directory / filename) for filename in files})
    temporary = directory / 'evidence.zip.tmp'
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for filename in [*files, 'sha256.json']:
            archive.write(directory / filename, filename)
    os.replace(temporary, directory / 'evidence.zip')


def _finish(directory, config, state, client, *, reload_proof=None, log_completion=False):
    """Rebuild reports; finalization errors fail visibly without losing checkpoints."""
    try:
        _write_json(directory / 'config.json', config.to_dict())
        _write_json(directory / 'data_identity.json', state['data_identity'])
        latest = best = None
        references = {}
        if (directory / 'latest.pt').exists():
            latest, best, references = _export_checkpoints(directory, config, state['data_identity'])
        history = [] if latest is None else latest['history']
        _sync_tracking(client, directory, state, config, history)
        state['epochs_completed'] = len(history)
        _write_json(directory / 'run_state.json', state)
        summary = {'status': state['status'], 'stop_reason': None if latest is None else latest['stop_reason'],
                   'epochs_completed': len(history), 'optimizer_updates': 0 if latest is None else latest['optimizer_updates'],
                   'skipped_updates': sum(row['skipped_updates'] for row in history), 'history': history,
                   'best_epoch': None if latest is None else latest['best_epoch'],
                   'best_metrics': None if best is None else {key: value for key, value in best['validation'].items()
                                                             if type(value) in (int, float)},
                   'latest_metrics': None if latest is None else {key: value for key, value in latest['validation'].items()
                                                                 if type(value) in (int, float)},
                   'runtime': state['runtime'], 'device': state['device'],
                   'resolved_precision': state['resolved_precision'], 'started_at': state['started_at'],
                   'elapsed_seconds': state['elapsed_seconds'], 'data_identity': state['data_identity'],
                   'weight_provenance': state.get('weight_provenance'), 'checkpoints': references,
                   'checkpoint_reload': reload_proof or {'status': 'not_executed'},
                   'tracking': state['tracking'], 'failures': state['failures'],
                   'limitations': ['One writer per run directory; recovery is at completed epochs.',
                                   'No official test images accessed. Genuine GPU acceptance requires the user-run notebook.']}
        _write_json(directory / 'run_summary.json', summary)
        if client is not None and state['tracking']['status'] != 'error':
            try:
                client.log_artifact(state['tracking']['run_id'], str(directory / 'run_summary.json'))
            except Exception as error:
                _tracking_error(directory, state, error, client)
                _write_json(directory / 'run_state.json', state)
                _write_json(directory / 'run_summary.json', summary)
        if log_completion:
            _log(directory, 'Training completed; best checkpoint reloaded and forward verified.')
        _evidence(directory)
        return summary

    except (Exception, KeyboardInterrupt) as error:
        # Do not call this finalizer again: report IO itself may be unavailable.
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['failures'].append(_failure('finalization', error))
        if client is not None:
            state['tracking']['status'] = 'failed'
            try:
                client.set_terminated(state['tracking']['run_id'], status='FAILED')
            except Exception as tracking_error:
                state['failures'].append(_failure('tracking', tracking_error))
                state['tracking']['status'] = 'error'
        try:
            summary = json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))
            if not isinstance(summary, dict):
                summary = {}
        except (OSError, UnicodeError, ValueError):
            summary = {}
        summary.update(status=state['status'], epochs_completed=state.get('epochs_completed', 0),
                       failures=state['failures'], tracking=state['tracking'])
        for filename, report in (('run_state.json', state), ('run_summary.json', summary)):
            try:
                _write_json(directory / filename, report)
            except Exception:
                pass  # Preserve the original error even if failure recording fails.
        raise


def run_training(prepared_dir, run_dir, config, *, device='cuda', resume=False,
                 tracking='mlflow', source=None) -> dict:
    """Run or resume classification using canonical metadata and pinned weights.

    config accepts TrainConfig or its plain JSON mapping. Terminal resume needs
    no prepared data/source pixels: checkpoint config and identity are checked
    first, then saved portable results are regenerated without training.
    """
    config = config if isinstance(config, TrainConfig) else TrainConfig.from_dict(config)
    if config.seed != 42:
        raise ValueError('Canonical training seed must be 42')
    if tracking not in ('json', 'mlflow'):
        raise ValueError('tracking must be mlflow or json')
    if type(resume) is not bool:
        raise TypeError('resume must be bool')
    device = torch.device(device)
    directory = Path(run_dir).resolve()
    prepared = Path(prepared_dir).resolve()
    if directory == prepared or prepared in directory.parents or directory in prepared.parents:
        raise ValueError('Prepared and run directories must be separate without nesting')
    if not resume and directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError('Use a fresh run directory or explicit resume=True')
    saved = None
    if resume:
        saved = read_checkpoint(directory / 'latest.pt', expected_config=config)
        state = json.loads((directory / 'run_state.json').read_text(encoding='utf-8'))
        if saved['data_identity'] != state['data_identity']:
            raise ValueError('Run state and checkpoint data identity mismatch')
        if tracking != state['tracking']['mode']:
            raise ValueError('Resume tracking mode must match saved run')
        state['weight_provenance'] = saved['weight_provenance']
        _validated_best(directory, config, saved)
        # Validate both retained checkpoints before opening source pixels.
        best_model, _ = load_model_checkpoint(directory / 'best.pt', device='cpu',
                                              expected_data_identity=state['data_identity'])
        del best_model
        if saved['stop_reason'] is not None:
            proof = {'status': 'state_loaded_without_forward', 'reason': 'Terminal recovery did not access source pixels.'}
            try:
                old = json.loads((directory / 'run_summary.json').read_text(encoding='utf-8'))
                digest = _sha256(directory / 'best.pt')
                candidate = old.get('checkpoint_reload', {})
                if (old.get('checkpoints', {}).get('best', {}).get('sha256') == digest
                        and isinstance(candidate, dict) and candidate.get('status') == 'forward_verified'
                        and candidate.get('checkpoint_sha256') == digest):
                    proof = candidate
            except (OSError, UnicodeError, ValueError, TypeError, AttributeError):
                pass  # Optional reports never prevent checkpoint-based recovery.
            state['status'] = 'completed'
            return _finish(directory, config, state, _open_tracking(directory, state), reload_proof=proof)
    precision = resolve_precision(config.amp, device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    identity = validate_preflight_preparation(prepared)
    if saved is not None and saved['data_identity'] != identity:
        raise ValueError('Checkpoint data identity mismatch')
    if saved is not None and saved['resolved_precision'] != precision:
        raise ValueError('Resolved precision differs from checkpoint')
    if saved is None:
        state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': 0.0, 'device': str(device), 'resolved_precision': precision,
                 'runtime': collect_runtime(), 'data_identity': identity, 'failures': [],
                 'tracking': {'mode': tracking, 'status': 'disabled' if tracking == 'json' else 'pending',
                              'run_id': None, 'location': 'mlruns' if tracking == 'mlflow' else None}}
    state['status'] = 'running'
    state['device'] = str(device)
    # Import requested tracking before opening pixels or constructing a model.
    directory.mkdir(parents=True, exist_ok=True)
    client = _open_tracking(directory, state)
    _write_json(directory / 'run_state.json', state)
    _write_json(directory / 'config.json', config.to_dict())
    _write_json(directory / 'data_identity.json', identity)
    started = time.monotonic()
    try:
        model_config = ModelConfig(config.architecture, fine_tuning=config.phases[0].mode,
                                   head_dropout=config.head_dropout, freeze_bn_stats=config.freeze_bn_stats)
        if saved is not None:
            model, _ = load_model_checkpoint(directory / 'latest.pt', device=device, expected_data_identity=identity)
        loaders = make_pretrained_loaders(prepared, model_config,
                                         preprocessing=PreprocessingConfig(config.image_size, config.normalization),
                                         source=source, batch_size=config.batch_size, num_workers=config.num_workers,
                                         seed=config.seed, pin_memory=device.type == 'cuda')
        if set(loaders) != {'train', 'val'} or any(len(loaders[name].dataset) != count
                                                 for name, count in (('train', 4794), ('val', 1200))):
            raise ValueError('Expected only train/val loaders with dataset lengths 4794/1200')
        if saved is None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            model, provenance = build_verified_pretrained_model(model_config)
            state['weight_provenance'] = provenance
        else:
            provenance = saved['weight_provenance']
        _write_json(directory / 'run_state.json', state)

        def completed_epoch(record, validation):
            _check_counts(record)
            _log(directory, f"Epoch {record['epoch']}: train_loss={record['train_loss']:.6f}, "
                 f"val_loss={record['val_loss']:.6f}, updates={record['optimizer_updates']}")

        fit(model, loaders, config, directory, data_identity=identity, weight_provenance=provenance,
            device=device, resume=resume, on_epoch=completed_epoch)
        del model
        best_model, _ = load_model_checkpoint(directory / 'best.pt', device=device, expected_data_identity=identity)
        batch = next(iter(loaders['val']))
        with torch.inference_mode():
            logits = best_model(batch['image'].to(device))
        if tuple(logits.shape) != (len(batch['label']), 200) or not torch.isfinite(logits).all():
            raise ValueError('Best checkpoint reload produced invalid 200-class logits')
        proof = {'status': 'forward_verified', 'logits_shape': list(logits.shape), 'finite': True,
                 'checkpoint_sha256': _sha256(directory / 'best.pt')}
        state['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        state['failures'].append(_failure('training', error))
        state['elapsed_seconds'] += time.monotonic() - started
        try:
            _write_json(directory / 'run_state.json', state)
            _log(directory, f"{state['status']}: {state['failures'][-1]['message']}")
        except Exception:
            pass
        try:
            _finish(directory, config, state, client)
        except (Exception, KeyboardInterrupt):
            pass  # The finalizer records its own failure; preserve the training error.
        raise
    state['elapsed_seconds'] += time.monotonic() - started
    return _finish(directory, config, state, client, reload_proof=proof, log_completion=True)
