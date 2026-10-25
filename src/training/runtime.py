"""Official-weight GPU smoke checks and portable evidence, without training updates.

CPU execution validates metadata and exports pending evidence only. The private
batch primitive also supports CPU arithmetic tests; it does not certify a GPU.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from torch import nn

from src.data.preparation import load_classification_manifest
from src.data.source import CubSource, _SOURCES, load_cub_split, validate_source_schema
from src.models.pretrained import ModelConfig
from src.training.data import PreprocessingConfig, make_pretrained_loaders
from src.training.weights import build_verified_pretrained_model


SUPPORTED_ARCHITECTURES = ('convnext_tiny', 'efficientnet_b3', 'regnet_y_3_2gf')
CANONICAL_MANIFEST_SHA256 = 'a8bd05e048febf32df6d84a6e1bc16eddf6416be723e12dfc89621044962736b'
_PARTITION_COUNTS = {'train': 4794, 'val': 1200, 'test': 5794, 'excluded': 0}


def _separate(first, second):
    return first != second and first not in second.parents and second not in first.parents


@dataclass(frozen=True)
class PreflightConfig:
    prepared_dir: str | Path
    output_dir: str | Path
    persistent_dir: str | Path | None = None
    architectures: tuple[str, ...] = SUPPORTED_ARCHITECTURES
    batch_size: int = 2
    num_workers: int = 0
    seed: int = 42
    offline: bool = False
    warmup_batches: int = 2
    measured_batches: int = 5
    amp: str = 'auto'

    def __post_init__(self):
        for key in ('prepared_dir', 'output_dir', 'persistent_dir'):
            value = getattr(self, key)
            if value is None and key != 'persistent_dir':
                raise ValueError(f'{key} must be an explicit path')
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f'{key} must be an explicit path')
                object.__setattr__(self, key, Path(value).resolve())
        paths = [self.prepared_dir, self.output_dir]
        if self.persistent_dir is not None:
            paths.append(self.persistent_dir)
        if any(not _separate(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]):
            raise ValueError('Preparation, output and persistent paths must be separate, without nesting')
        if (not isinstance(self.architectures, (tuple, list)) or not self.architectures
                or any(name not in SUPPORTED_ARCHITECTURES for name in self.architectures)
                or len(set(self.architectures)) != len(self.architectures)):
            raise ValueError('architectures must be a nonempty unique supported sequence')
        object.__setattr__(self, 'architectures', tuple(self.architectures))
        for key, minimum in (('batch_size', 1), ('num_workers', 0), ('warmup_batches', 1), ('measured_batches', 1)):
            if type(getattr(self, key)) is not int or getattr(self, key) < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')
        if type(self.seed) is not int or self.seed != 42:
            raise ValueError('Canonical preflight seed must be 42')
        if type(self.offline) is not bool:
            raise TypeError('offline must be bool')
        if self.amp not in ('auto', 'float16', 'bfloat16', 'off'):
            raise ValueError('amp must be auto, float16, bfloat16 or off')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_runtime(repo_dir=None):
    """Allowlisted facts only: no environment variables, remotes or credentials."""
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'datasets', 'fsspec', 'huggingface-hub'):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    cuda = {'available': bool(torch.cuda.is_available()), 'torch_cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(), 'devices': []}
    if cuda['available']:
        try:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                cuda['devices'].append({'index': index, 'name': properties.name,
                                        'total_memory_bytes': properties.total_memory,
                                        'compute_capability': [properties.major, properties.minor]})
            cuda['selected_device_index'] = torch.cuda.current_device()
            cuda['bfloat16_supported'] = bool(torch.cuda.is_bf16_supported())
        except Exception as error:
            cuda['inspection_failure'] = _failure('runtime', error)
    git = {'status': 'unavailable'}
    try:
        directory = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[2]
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=directory,
                                capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        git = {'status': 'partial', 'commit': commit, 'dirty': None}
        dirty = subprocess.run(['git', '--no-optional-locks', 'status', '--porcelain'], cwd=directory,
                               capture_output=True, text=True, check=True, timeout=10).stdout
        git = {'status': 'available', 'commit': commit, 'dirty': bool(dirty.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'versions': versions, 'cuda': cuda, 'git': git}


def validate_preflight_preparation(prepared_dir):
    """Validate canonical metadata with the existing full validator before HF I/O."""
    directory = Path(prepared_dir)
    rows = load_classification_manifest(directory)
    summary = json.loads((directory / 'preparation.json').read_text(encoding='utf-8'))
    if (summary['manifest_sha256'] != CANONICAL_MANIFEST_SHA256
            or summary['seed'] != 42 or summary['partition_counts'] != _PARTITION_COUNTS):
        raise ValueError('Preparation does not match the canonical manifest SHA, seed42 and partition counts')
    if len(summary['class_names']) != 200 or len(set(summary['class_names'])) != 200:
        raise ValueError('Canonical ordered 200-class mapping required')
    if len(rows) != sum(_PARTITION_COUNTS.values()):
        raise ValueError('Canonical manifest row count mismatch')
    class_bytes = json.dumps(summary['class_names'], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return {'status': 'metadata_validated', 'manifest_sha256': summary['manifest_sha256'],
            'preparation_sha256': _sha256(directory / 'preparation.json'),
            'class_mapping_sha256': hashlib.sha256(class_bytes).hexdigest(),
            'class_mapping_hash_encoding': 'UTF-8 compact JSON ordered class_names, ensure_ascii=False',
            'class_names': summary['class_names'], 'partition_counts': summary['partition_counts'],
            'seed': summary['seed'], 'source_revisions': summary['source_revisions'],
            'official_test_images_accessed': False, 'source_status': 'not_run'}


def _load_train_source(*, offline):
    if not offline:
        return load_cub_split('train')
    # DownloadConfig alone can still allow Hub metadata requests in datasets.
    # Resolve only an already-staged, exact-revision raw snapshot, then use a
    # local dataset path. An Arrow cache alone is not this offline contract.
    from datasets import DownloadConfig, Image, load_dataset
    from huggingface_hub import snapshot_download
    spec = _SOURCES['train']
    snapshot = snapshot_download(repo_id=spec.repo_id, repo_type='dataset',
                                 revision=spec.revision, local_files_only=True)
    dataset = load_dataset(snapshot, split=spec.source_split,
                           download_config=DownloadConfig(local_files_only=True))
    classes = validate_source_schema(dataset.features)
    return CubSource(spec, dataset.cast_column('image', Image(decode=False)), classes)


def _failure(stage, error):
    message = str(error)
    message = re.sub(r'(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+', '[REDACTED]', message)
    message = re.sub(r'https?://\S+', '[URL omitted]', message)
    return {'stage': stage, 'type': type(error).__name__, 'message': message[:1500]}


def _memory(device):
    if device.type != 'cuda':
        return {'status': 'not_cuda'}
    torch.cuda.synchronize(device)
    return {'status': 'measured', 'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def _gradient_report(model, policy):
    named = list(model.named_parameters())
    active = [(name, p) for name, p in named if p.requires_grad]
    frozen = [(name, p) for name, p in named if not p.requires_grad]
    if any(p.grad is not None for _, p in frozen):
        raise RuntimeError('Frozen parameters received gradients')
    if not active or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _, p in active):
        raise RuntimeError('Active gradients must all be present and finite after unscaling; AMP overflow is a failure')
    nonzero = [name for name, p in active if bool(torch.count_nonzero(p.grad))]
    if not any(name.startswith('head.') for name in nonzero):
        raise RuntimeError('Head has no meaningful nonzero gradient')
    encoder_active = [name for name, _ in active if not name.startswith('head.')]
    encoder_frozen = [name for name, _ in frozen if not name.startswith('head.')]
    if ((policy == 'frozen' and encoder_active)
            or (policy == 'partial' and (not encoder_active or not encoder_frozen))
            or (policy == 'full' and encoder_frozen)):
        raise RuntimeError('Parameter mask disagrees with the fine-tuning policy')
    if policy != 'frozen' and not any(not name.startswith('head.') for name in nonzero):
        raise RuntimeError('Active encoder has no meaningful nonzero gradient')
    return {'active_parameter_tensors': len(active), 'frozen_parameter_tensors': len(frozen),
            'nonzero_gradient_tensors': len(nonzero), 'all_active_present_finite': True,
            'frozen_with_gradients': 0, 'checked_after_unscaling': True,
            'active_encoder_parameter_tensors': len(encoder_active)}


def _output_summary(model, images, features, logits, loss):
    channels = model.spec.encoder_channels
    # The fixed 224-pixel preflight input is divisible by every supported
    # encoder's stride (32), so its retained spatial grid must be exactly 7x7.
    spatial_shape = [len(images), channels,
                     images.shape[-2] // model.spec.output_stride,
                     images.shape[-1] // model.spec.output_stride]
    if (list(logits.shape) != [len(images), 200]
            or list(features['pooled'].shape) != [len(images), channels]
            or list(features['spatial'].shape) != spatial_shape):
        raise RuntimeError('Logits/spatial/pooled output shapes violate the adapter contract')
    if any(not bool(torch.isfinite(value).all()) for value in (logits, loss, *features.values())):
        raise RuntimeError('Forward outputs and loss must be finite')
    return {'loss': float(loss.detach().item()),
            'shapes': {'images': list(images.shape), 'logits': list(logits.shape),
                       'spatial': list(features['spatial'].shape), 'pooled': list(features['pooled'].shape)}}


def _check_batch(model, batch, *, policy, precision, device):
    """One loss/backward only, clearing every gradient on both success and error."""
    model.zero_grad(set_to_none=True)
    model.set_fine_tuning(policy)
    model.train()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = scaler = None
    try:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(precision)
        context = torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()
        with context:
            features = model.forward_features(images)
            logits = model.head(features['pooled'])
            loss = nn.functional.cross_entropy(logits, labels)
        outputs = _output_summary(model, images, features, logits, loss)
        if precision == 'float16':
            # This optimizer exists ONLY for public GradScaler.unscale_. No
            # optimizer/scaler step or weight update is ever performed.
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
            scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        gradients = _gradient_report(model, policy)
        return {'status': 'passed', 'policy': policy, 'precision': precision,
                'execution_device': str(device), 'mode': 'train', 'backward': 'passed', **outputs,
                'gradients': gradients, 'memory': _memory(device), 'optimizer_updates': 0,
                'grad_scaler_scale': None if scaler is None else scaler.get_scale(),
                'amp_overflow': False}
    finally:
        model.zero_grad(set_to_none=True)
        del optimizer, scaler


def _check_validation(model, batch, *, policy, precision, device):
    """Validation checks are evaluation-only, without an autograd graph."""
    model.zero_grad(set_to_none=True)
    model.set_fine_tuning(policy)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(precision)
        context = torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()
        with torch.inference_mode(), context:
            features = model.forward_features(images)
            logits = model.head(features['pooled'])
            loss = nn.functional.cross_entropy(logits, labels)
            outputs = _output_summary(model, images, features, logits, loss)
        return {'status': 'passed', 'policy': policy, 'precision': precision,
                'execution_device': str(device), 'mode': 'eval', 'backward': 'not_run', **outputs,
                'gradients': {'status': 'not_run', 'reason': 'Validation is inference only'},
                'memory': _memory(device), 'optimizer_updates': 0}
    finally:
        model.zero_grad(set_to_none=True)


def _throughput(model, batch, config, device):
    model.eval()
    images = batch['image'].to(device)
    with torch.inference_mode():
        for _ in range(config.warmup_batches):
            model(images)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(config.measured_batches):
            model(images)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    return {'status': 'measured', 'measurement': 'warmed synchronized GPU inference on resident validation batch',
            'precision': 'fp32', 'includes_data_loading': False, 'includes_backward': False,
            'is_training_runtime': False, 'batch_size': len(images), 'execution_device': str(device),
            'warmup_batches': config.warmup_batches, 'measured_batches': config.measured_batches,
            'elapsed_seconds': elapsed, 'images_per_second': len(images) * config.measured_batches / elapsed}


def _run_candidate(config, architecture, batches):
    result = {'architecture': architecture, 'status': 'not_run', 'weight_verification': {'status': 'not_run'},
              'checks': [], 'throughput': {'status': 'not_run'}, 'failures': []}
    model = None
    device = torch.device('cuda', torch.cuda.current_device())
    stage = 'weights'
    try:
        model_config = ModelConfig(architecture)
        model, proof = build_verified_pretrained_model(model_config, offline=config.offline, progress=True)
        result['weight_verification'] = {'status': 'verified', **proof}
        result['preprocessing'] = PreprocessingConfig().describe(model_config)
        stage = 'device_transfer'
        model.to(device)
        precisions = ['fp32']
        if config.amp == 'off':
            result['checks'].append({'precision': 'amp', 'status': 'not_run', 'reason': 'explicitly disabled'})
        elif config.amp == 'bfloat16' and not torch.cuda.is_bf16_supported():
            result['checks'].append({'precision': 'bfloat16', 'status': 'unsupported', 'reason': 'CUDA device does not support bfloat16'})
        else:
            precisions.append(('bfloat16' if torch.cuda.is_bf16_supported() else 'float16') if config.amp == 'auto' else config.amp)
        stage = 'smoke_checks'
        for policy in ('frozen', 'partial', 'full'):
            for precision in precisions:
                for partition, batch in batches.items():
                    try:
                        check_function = _check_batch if partition == 'train' else _check_validation
                        check = check_function(model, batch, policy=policy, precision=precision, device=device)
                        check['partition'] = partition
                        result['checks'].append(check)
                    except Exception as error:
                        failure = _failure(stage, error)
                        result['checks'].append({'policy': policy, 'precision': precision, 'partition': partition,
                                                 'status': 'failed', 'failure': failure, 'memory': _memory(device)})
                        result['failures'].append(failure)
                    finally:
                        model.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
        stage = 'throughput'
        result['throughput'] = _throughput(model, batches['val'], config, device)
        result['status'] = ('failed' if result['failures'] else
                            'incomplete' if any(check['status'] != 'passed' for check in result['checks']) else 'passed')
    except Exception as error:
        result['status'] = 'failed'
        result['failures'].append(_failure(stage, error))
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _write_json(path, value, *, mode='x'):
    with path.open(mode, encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _copy_evidence(source, target):
    with target.open('xb') as destination, source.open('rb') as origin:
        shutil.copyfileobj(origin, destination)
    if _sha256(source) != _sha256(target):
        raise OSError('Persistent evidence artifact readback hash mismatch')


def _export(report, config, *, replace_current=False):
    output = config.output_dir
    report['artifacts'] = {'report': str(output / 'preflight-report.json'),
                           'inventory': str(output / 'evidence-inventory.json'),
                           'evidence_zip': str(output / 'preflight-evidence.zip')}
    report_path = Path(report['artifacts']['report'])
    mode = 'w' if replace_current else 'x'
    _write_json(report_path, report, mode=mode)
    inventory = {'schema_version': 1, 'hash_algorithm': 'sha256',
                 'files': [{'name': report_path.name, 'sha256': _sha256(report_path), 'size_bytes': report_path.stat().st_size}],
                 'exclusions': ['weights', 'images', 'credentials', 'environment dumps'],
                 'inventory_self_hash': 'not included to avoid recursive hashing'}
    _write_json(Path(report['artifacts']['inventory']), inventory, mode=mode)
    with ZipFile(report['artifacts']['evidence_zip'], mode, compression=ZIP_DEFLATED) as archive:
        for key in ('report', 'inventory'):
            path = Path(report['artifacts'][key])
            archive.write(path, path.name)
    if config.persistent_dir is not None and report['persistence']['status'] == 'write_read_verified':
        try:
            for path in report['artifacts'].values():
                source = Path(path)
                target = config.persistent_dir / source.name
                # Persistent directory was exclusively created for this run.
                _copy_evidence(source, target)
        except Exception as error:
            report['persistence']['status'] = 'failed'
            report['status'] = 'failed'
            report['acceptance_complete'] = False
            if 'persistent_storage' not in report['acceptance_pending']:
                report['acceptance_pending'].append('persistent_storage')
            report['failures'].append(_failure('persistence_export', error))
            # Refresh only artifacts just created in our exclusively reserved
            # run directory. Previous runs are never overwritten.
            _export(report, config, replace_current=True)


def run_gpu_preflight(config):
    """Export partial failures and CPU-pending state; only CUDA executes models."""
    if not isinstance(config, PreflightConfig):
        raise TypeError('config must be a PreflightConfig')
    for directory in (config.output_dir, config.persistent_dir):
        if directory is not None and directory.exists():
            raise FileExistsError(f'Use a fresh run directory; refusing to overwrite {directory}')
    config.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'not_run', 'runtime': collect_runtime(), 'dataset': {'status': 'not_run'},
              'coverage': {'supported': list(SUPPORTED_ARCHITECTURES), 'requested': list(config.architectures),
                           'all_supported_requested': set(config.architectures) == set(SUPPORTED_ARCHITECTURES)},
              'configuration': {'batch_size': config.batch_size, 'num_workers': config.num_workers,
                                'seed': config.seed, 'offline': config.offline, 'amp': config.amp,
                                'warmup_batches': config.warmup_batches, 'measured_batches': config.measured_batches},
              'models': [{'architecture': name, 'status': 'not_run', 'weight_verification': {'status': 'not_run'},
                          'checks': [], 'throughput': {'status': 'not_run'}, 'failures': []} for name in config.architectures],
              'failures': [], 'persistence': {'status': 'not_requested', 'fresh_session_durability': 'not_tested',
                                             'limitation': 'Same-session write/read cannot prove persistence after a runtime restart.'}}
    stage = 'preparation'
    try:
        report['dataset'] = validate_preflight_preparation(config.prepared_dir)
        if 'inspection_failure' in report['runtime']['cuda']:
            stage = 'runtime'
            raise RuntimeError(report['runtime']['cuda']['inspection_failure']['message'])
        if not report['runtime']['cuda']['available']:
            report['status'] = 'pending_gpu'
        else:
            stage = 'source'
            source = _load_train_source(offline=config.offline)
            if tuple(report['dataset']['class_names']) != source.class_names:
                raise ValueError('Pinned source ordered class mapping differs from canonical preparation')
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            loaders = make_pretrained_loaders(config.prepared_dir, ModelConfig(config.architectures[0]),
                source=source, batch_size=config.batch_size, num_workers=config.num_workers,
                seed=config.seed, pin_memory=True)
            batches = {partition: next(iter(loaders[partition])) for partition in ('train', 'val')}
            report['dataset']['source_status'] = 'representative_train_val_loaded'
            report['dataset']['representative_batches'] = {
                partition: {'image_ids': batch['image_id'], 'labels': batch['label'].tolist(),
                            'shape': list(batch['image'].shape)} for partition, batch in batches.items()}
            stage = 'candidates'
            for index, architecture in enumerate(config.architectures):
                result = _run_candidate(config, architecture, batches)
                report['models'][index] = result
                report['failures'].extend({'architecture': architecture, **failure} for failure in result['failures'])
            statuses = [model['status'] for model in report['models']]
            report['status'] = 'failed' if 'failed' in statuses else 'passed' if all(s == 'passed' for s in statuses) else 'incomplete'
    except Exception as error:
        report['status'] = 'failed'
        report['failures'].append(_failure(stage, error))
    if config.persistent_dir is not None:
        try:
            config.persistent_dir.mkdir(parents=True, exist_ok=False)
            probe = config.persistent_dir / 'storage-readback.json'
            payload = {'schema_version': 1, 'created_utc': report['created_utc'], 'purpose': 'same-session storage write/read probe'}
            _write_json(probe, payload)
            if json.loads(probe.read_text(encoding='utf-8')) != payload:
                raise OSError('Storage probe readback mismatch')
            report['persistence'].update(status='write_read_verified', probe_sha256=_sha256(probe),
                                         persistent_dir=str(config.persistent_dir))
        except Exception as error:
            report['persistence']['status'] = 'failed'
            report['status'] = 'failed'
            report['failures'].append(_failure('persistence', error))
    pending = []
    if not report['runtime']['cuda']['available']:
        pending.append('actual_cuda_execution')
    if not report['coverage']['all_supported_requested']:
        pending.append('all_supported_architectures')
    if config.amp == 'off':
        pending.append('amp_checks')
    if report['persistence']['status'] != 'write_read_verified':
        pending.append('persistent_storage')
    if report['status'] != 'passed':
        pending.append('all_requested_model_checks')
    report['acceptance_pending'] = pending
    report['acceptance_complete'] = not pending
    _export(report, config)
    return report
