"""Offline CPU contracts; mocks never establish executed cloud/GPU evidence."""
import contextlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
import torch
from torch import nn


def _warn_if_cpu():
    if not torch.cuda.is_available():
        return pytest.warns(UserWarning, match='pin_memory')
    return contextlib.nullcontext()


def api():
    return importlib.import_module('src.training.runtime')


@pytest.fixture
def canonical_fixture(monkeypatch, audited_prepared):
    runtime = api()
    prepared = audited_prepared[0]
    summary = json.loads((prepared / 'preparation.json').read_text())
    # A valid synthetic preparation exercises the real validator; it is never
    # presented as the production canonical preparation or official evidence.
    monkeypatch.setattr(runtime, 'CANONICAL_MANIFEST_SHA256', summary['manifest_sha256'])
    return prepared


@pytest.mark.parametrize('kwargs', [
    {'batch_size': 0}, {'batch_size': True}, {'num_workers': -1},
    {'seed': 1}, {'warmup_batches': 0}, {'measured_batches': False},
    {'amp': 'fp8'}, {'architectures': ()}, {'architectures': ('resnet50',)},
    {'architectures': ('convnext_tiny', 'convnext_tiny')}, {'offline': 1},
])
def test_config_rejects_invalid_values(tmp_path, kwargs):
    with pytest.raises((ValueError, TypeError)):
        api().PreflightConfig(tmp_path / 'prepared', tmp_path / 'output', **kwargs)


@pytest.mark.parametrize('placement', ['same', 'inside', 'parent'])
def test_output_cannot_overwrite_preparation(tmp_path, placement):
    prepared = tmp_path / 'prepared'
    output = {'same': prepared, 'inside': prepared / 'run', 'parent': tmp_path}[placement]
    with pytest.raises(ValueError, match='separate'):
        api().PreflightConfig(prepared, output)


def test_cpu_pending_exports_without_loading_source_or_weights(canonical_fixture, tmp_path, monkeypatch):
    runtime = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: pytest.fail('CPU source fetch'))
    monkeypatch.setattr(runtime, 'build_verified_pretrained_model', lambda *a, **kw: pytest.fail('CPU weights'))
    report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'run', persistent_dir=tmp_path / 'persistent'))
    assert report['status'] == 'pending_gpu'
    assert report['dataset']['status'] == 'metadata_validated'
    assert report['dataset']['partition_counts'] == {'train': 4794, 'val': 1200, 'test': 5794, 'excluded': 0}
    assert len(report['dataset']['class_names']) == 200
    assert all(model['status'] == 'not_run' and model['weight_verification']['status'] == 'not_run' for model in report['models'])
    assert report['persistence']['status'] == 'write_read_verified'
    assert report['persistence']['fresh_session_durability'] == 'not_tested'
    path = Path(report['artifacts']['report'])
    assert json.loads(path.read_text()) == report
    inventory = json.loads(Path(report['artifacts']['inventory']).read_text())
    assert inventory['files'][0]['sha256'] == runtime._sha256(path)
    with ZipFile(report['artifacts']['evidence_zip']) as archive:
        assert set(archive.namelist()) == {'preflight-report.json', 'evidence-inventory.json'}
    assert (tmp_path / 'persistent' / path.name).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'run'))


def test_production_hash_rejects_synthetic_before_source(audited_prepared, tmp_path, monkeypatch):
    runtime = api()
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: pytest.fail('source fetched before validation'))
    report = runtime.run_gpu_preflight(runtime.PreflightConfig(audited_prepared[0], tmp_path / 'invalid'))
    assert report['status'] == 'failed'
    assert 'canonical' in report['failures'][0]['message'].lower()
    assert all(model['status'] == 'not_run' for model in report['models'])


def test_missing_preparation_preserves_failure_report(tmp_path):
    report = api().run_gpu_preflight(api().PreflightConfig(tmp_path / 'absent', tmp_path / 'failed'))
    assert report['status'] == 'failed'
    assert Path(report['artifacts']['evidence_zip']).is_file()


def test_runtime_is_allowlisted(monkeypatch):
    monkeypatch.setenv('EXAMPLE_SECRET', 'do-not-export')
    facts = api().collect_runtime()
    assert 'do-not-export' not in json.dumps(facts)
    assert set(facts) == {'python', 'platform', 'versions', 'cuda', 'git'}
    assert isinstance(facts['cuda']['available'], bool)
    assert 'fsspec' in facts['versions']
    assert set(facts['git']) <= {'commit', 'dirty', 'status'}


class TinyAdapter(nn.Module):
    """Small CPU fixture for arithmetic/gradient contracts, not an architecture."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Conv2d(4, 4, 1))
        self.encoder_norm = nn.Identity()
        self.head = nn.Linear(4, 200)
        self.spec = SimpleNamespace(encoder_channels=4, output_stride=1)

    def set_fine_tuning(self, policy):
        self.encoder.requires_grad_(policy == 'full')
        if policy == 'partial':
            self.encoder[2].requires_grad_(True)
        self.head.requires_grad_(True)

    def forward_features(self, images):
        spatial = self.encoder(images)
        return {'spatial': spatial, 'pooled': spatial.mean((2, 3))}

    def forward(self, images):
        return self.head(self.forward_features(images)['pooled'])


@pytest.mark.parametrize('policy', ['frozen', 'partial', 'full'])
def test_cpu_smoke_primitive_has_finite_gradients_without_updates(policy):
    runtime = api()
    torch.manual_seed(42)
    model = TinyAdapter()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = {'image': torch.randn(2, 3, 8, 8), 'label': torch.tensor([0, 1])}
    result = runtime._check_batch(model, batch, policy=policy, precision='fp32', device=torch.device('cpu'))
    assert result['execution_device'] == 'cpu' and result['status'] == 'passed'
    assert result['shapes'] == {'images': [2, 3, 8, 8], 'logits': [2, 200], 'spatial': [2, 4, 8, 8], 'pooled': [2, 4]}
    assert result['gradients']['all_active_present_finite'] is True
    assert result['gradients']['frozen_with_gradients'] == 0
    assert result['optimizer_updates'] == 0
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cpu_smoke_rejects_nonfinite_and_cleans_gradients():
    model = TinyAdapter()
    batch = {'image': torch.full((2, 3, 8, 8), float('nan')), 'label': torch.tensor([0, 1])}
    with pytest.raises(RuntimeError, match='finite'):
        api()._check_batch(model, batch, policy='full', precision='fp32', device=torch.device('cpu'))
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize('spatial_shape', [(2, 4, 6, 7), (2, 4, 7, 8)])
def test_output_contract_rejects_incorrect_spatial_geometry(spatial_shape):
    model = TinyAdapter()
    model.spec.output_stride = 32
    images = torch.randn(2, 3, 224, 224)
    features = {'spatial': torch.randn(spatial_shape), 'pooled': torch.randn(2, 4)}
    logits = torch.randn(2, 200)
    loss = nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    with pytest.raises(RuntimeError, match='shapes'):
        api()._output_summary(model, images, features, logits, loss)


def test_source_failure_is_exported_in_mocked_cuda_orchestration(canonical_fixture, tmp_path, monkeypatch):
    runtime = api()
    facts = runtime.collect_runtime()
    facts['cuda']['available'] = True
    monkeypatch.setattr(runtime, 'collect_runtime', lambda **kw: facts)
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: (_ for _ in ()).throw(RuntimeError('source unavailable')))
    report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'source-failure'))
    assert report['status'] == 'failed' and report['failures'][0]['stage'] == 'source'
    assert all(model['status'] == 'not_run' for model in report['models'])


def test_mocked_candidate_failures_preserve_later_coverage(canonical_fixture, tmp_path, monkeypatch, pinned_source):
    runtime = api()
    facts = runtime.collect_runtime()
    facts['cuda']['available'] = True
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.accelerator, 'is_available', lambda: False)
    monkeypatch.setattr(runtime, 'collect_runtime', lambda **kw: facts)
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: pinned_source)
    def candidate(config, architecture, batches):
        if architecture == 'convnext_tiny':
            return {'architecture': architecture, 'status': 'failed', 'failures': [{'stage': 'weights', 'type': 'RuntimeError', 'message': 'synthetic failure'}]}
        return {'architecture': architecture, 'status': 'passed', 'failures': []}
    monkeypatch.setattr(runtime, '_run_candidate', candidate)
    # The orchestration is simulated; loaders stay on CPU on every host.
    monkeypatch.setattr(torch.accelerator, 'is_available', lambda: False)
    with pytest.warns(UserWarning, match='pin_memory'):
        report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'mock-only'))
    assert report['status'] == 'failed'
    assert [model['status'] for model in report['models']] == ['failed', 'passed', 'passed']
    assert report['coverage']['requested'] == list(runtime.SUPPORTED_ARCHITECTURES)
    assert set(report['dataset']['representative_batches']) == {'train', 'val'}


def test_persistent_copy_failure_is_preserved_in_local_evidence(canonical_fixture, tmp_path, monkeypatch):
    runtime = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(runtime, '_copy_evidence', lambda *a, **kw: (_ for _ in ()).throw(OSError('storage disconnected')))
    report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'copy-failure', persistent_dir=tmp_path / 'remote'))
    assert report['status'] == 'failed'
    assert report['persistence']['status'] == 'failed'
    assert report['failures'][-1]['stage'] == 'persistence_export'
    assert json.loads(Path(report['artifacts']['report']).read_text()) == report
    assert report['acceptance_complete'] is False


def test_diagnostic_subset_and_disabled_amp_cannot_be_full_acceptance(canonical_fixture, tmp_path, monkeypatch, pinned_source):
    runtime = api()
    facts = runtime.collect_runtime()
    facts['cuda']['available'] = True
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.accelerator, 'is_available', lambda: False)
    monkeypatch.setattr(runtime, 'collect_runtime', lambda **kw: facts)
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: pinned_source)
    monkeypatch.setattr(runtime, '_run_candidate', lambda c, a, b: {'architecture': a, 'status': 'passed', 'failures': []})
    monkeypatch.setattr(torch.accelerator, 'is_available', lambda: False)
    with pytest.warns(UserWarning, match='pin_memory'):
        report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'diagnostic', architectures=('convnext_tiny',), amp='off'))
    assert report['status'] == 'passed'
    assert report['acceptance_complete'] is False
    assert set(report['acceptance_pending']) == {'all_supported_architectures', 'amp_checks', 'persistent_storage'}


def test_source_mapping_error_precedes_weight_access(canonical_fixture, tmp_path, monkeypatch, pinned_source):
    runtime = api()
    facts = runtime.collect_runtime()
    facts['cuda']['available'] = True
    monkeypatch.setattr(runtime, 'collect_runtime', lambda **kw: facts)
    pinned_source.class_names = tuple(reversed(pinned_source.class_names))
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: pinned_source)
    monkeypatch.setattr(runtime, 'build_verified_pretrained_model', lambda *a, **kw: pytest.fail('weights before source validation'))
    report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'wrong-source'))
    assert report['status'] == 'failed' and report['failures'][0]['stage'] == 'source'
    assert 'class mapping' in report['failures'][0]['message']


def test_existing_persistent_directory_is_never_modified(canonical_fixture, tmp_path):
    persistent = tmp_path / 'existing'
    persistent.mkdir()
    marker = persistent / 'old.json'
    marker.write_text('preserve')
    output = tmp_path / 'fresh'
    with pytest.raises(FileExistsError):
        api().run_gpu_preflight(api().PreflightConfig(canonical_fixture, output, persistent_dir=persistent))
    assert not output.exists() and marker.read_text() == 'preserve'



def test_offline_source_missing_snapshot_fails_without_dataset_network(monkeypatch):
    import datasets
    import huggingface_hub
    observed = {}
    def missing(**kwargs):
        observed.update(kwargs)
        raise FileNotFoundError('pinned snapshot not staged')
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', missing)
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **kw: pytest.fail('No snapshot: must not load remote data'))
    with pytest.raises(FileNotFoundError, match='not staged'):
        api()._load_train_source(offline=True)
    assert observed == {'repo_id': 'Multimodal-Fatima/CUB_train', 'repo_type': 'dataset',
                        'revision': 'bd3988995e6db193427255abdebd522afe24ce76', 'local_files_only': True}


def test_fp16_primitive_unscales_before_gradient_validation_without_step(monkeypatch):
    runtime = api()
    calls = []
    class FixtureScaler:
        def __init__(self, device, init_scale):
            assert device == 'cuda' and init_scale == 128.0
        def scale(self, loss):
            calls.append('scale')
            return loss * 128
        def unscale_(self, optimizer):
            calls.append('unscale')
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        parameter.grad.div_(128)
        def get_scale(self):
            return 128.0
    monkeypatch.setattr(torch.amp, 'GradScaler', FixtureScaler)
    monkeypatch.setattr(torch.optim.SGD, 'step', lambda *a, **kw: pytest.fail('Optimizer update forbidden'))
    original = runtime._gradient_report
    def checked(model, policy):
        calls.append('inspect')
        assert calls == ['scale', 'unscale', 'inspect']
        return original(model, policy)
    monkeypatch.setattr(runtime, '_gradient_report', checked)
    model = TinyAdapter()
    batch = {'image': torch.randn(2, 3, 8, 8), 'label': torch.tensor([0, 1])}
    result = runtime._check_batch(model, batch, policy='full', precision='float16', device=torch.device('cpu'))
    assert result['status'] == 'passed' and result['execution_device'] == 'cpu'
    assert result['grad_scaler_scale'] == 128.0 and result['optimizer_updates'] == 0


def test_mocked_candidate_records_check_failure_and_continues(tmp_path, monkeypatch):
    runtime = api()
    model = TinyAdapter()
    model.to = lambda device: model
    monkeypatch.setattr(runtime, 'build_verified_pretrained_model', lambda *a, **kw: (model, {'status': 'synthetic_fixture_only'}))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: False)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(runtime, '_memory', lambda device: {'status': 'mocked_not_evidence'})
    monkeypatch.setattr(runtime, '_throughput', lambda *a: {'status': 'mocked_not_evidence'})
    calls = []
    def check(model, batch, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError('synthetic overflow')
        return {'status': 'passed', 'policy': kwargs['policy'], 'precision': kwargs['precision']}
    monkeypatch.setattr(runtime, '_check_batch', check)
    monkeypatch.setattr(runtime, '_check_validation', check)
    config = runtime.PreflightConfig(tmp_path / 'prepared', tmp_path / 'output')
    result = runtime._run_candidate(config, 'convnext_tiny', {'train': {}, 'val': {}})
    assert len(calls) == 12 and result['status'] == 'failed'
    assert len(result['checks']) == 12 and len(result['failures']) == 1
    assert 'overflow' in result['failures'][0]['message']
    assert all(p.grad is None for p in model.parameters())


def test_fp16_nonfinite_unscaled_gradients_fail_and_clear(monkeypatch):
    class OverflowScaler:
        def __init__(self, *a, **kw):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        parameter.grad.fill_(float('inf'))
        def get_scale(self):
            return 128.0
    monkeypatch.setattr(torch.amp, 'GradScaler', OverflowScaler)
    model = TinyAdapter()
    batch = {'image': torch.randn(2, 3, 8, 8), 'label': torch.tensor([0, 1])}
    with pytest.raises(RuntimeError, match='finite.*unscaling'):
        api()._check_batch(model, batch, policy='full', precision='float16', device=torch.device('cpu'))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_checks_reject_wrong_parameter_mask():
    model = TinyAdapter()
    model.set_fine_tuning('full')
    model(torch.randn(2, 3, 8, 8)).sum().backward()
    with pytest.raises(RuntimeError, match='mask'):
        api()._gradient_report(model, 'frozen')


def test_persistence_paths_cannot_nest_output(tmp_path):
    with pytest.raises(ValueError, match='separate'):
        api().PreflightConfig(tmp_path / 'prepared', tmp_path / 'output', persistent_dir=tmp_path / 'output' / 'nested')


def test_explicit_unsupported_amp_cannot_be_acceptance(tmp_path, monkeypatch):
    runtime = api()
    model = TinyAdapter()
    model.to = lambda device: model
    monkeypatch.setattr(runtime, 'build_verified_pretrained_model', lambda *a, **kw: (model, {'status': 'synthetic_fixture_only'}))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: False)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(runtime, '_check_batch', lambda m, b, **kw: {'status': 'passed', 'precision': kw['precision']})
    monkeypatch.setattr(runtime, '_check_validation', lambda m, b, **kw: {'status': 'passed', 'precision': kw['precision']})
    monkeypatch.setattr(runtime, '_throughput', lambda *a: {'status': 'mocked_not_evidence'})
    config = runtime.PreflightConfig(tmp_path / 'prepared', tmp_path / 'output', amp='bfloat16')
    report = runtime._run_candidate(config, 'convnext_tiny', {'train': {}, 'val': {}})
    assert report['status'] == 'incomplete'
    assert report['checks'][0]['status'] == 'unsupported'
    assert len(report['checks']) == 7


def test_git_commit_is_retained_if_dirty_status_unavailable(monkeypatch):
    runtime = api()
    def git(command, **kwargs):
        if 'rev-parse' in command:
            return SimpleNamespace(stdout='a' * 40 + '\n')
        raise runtime.subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(runtime.subprocess, 'run', git)
    facts = runtime.collect_runtime()
    assert facts['git'] == {'status': 'partial', 'commit': 'a' * 40, 'dirty': None}


def test_cuda_inspection_failure_exports_without_loading_source(canonical_fixture, tmp_path, monkeypatch):
    runtime = api()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: (_ for _ in ()).throw(RuntimeError('GPU inspection failed')))
    monkeypatch.setattr(runtime, '_load_train_source', lambda **kw: pytest.fail('unusable CUDA must not load source'))
    report = runtime.run_gpu_preflight(runtime.PreflightConfig(canonical_fixture, tmp_path / 'cuda-error'))
    assert report['status'] == 'failed'
    assert report['failures'][0]['stage'] == 'runtime'
    assert report['runtime']['cuda']['available'] is True
    assert all(model['status'] == 'not_run' for model in report['models'])



def test_online_source_reuses_existing_train_only_loader(monkeypatch, pinned_source):
    runtime = api()
    calls = []
    def existing(split):
        calls.append(split)
        return pinned_source
    monkeypatch.setattr(runtime, 'load_cub_split', existing)
    assert runtime._load_train_source(offline=False) is pinned_source
    assert calls == ['train']


def test_validation_primitive_is_eval_inference_only_without_updates(monkeypatch):
    runtime = api()
    model = TinyAdapter()
    observed = []
    original = model.forward_features
    def features(images):
        observed.append((model.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return original(images)
    monkeypatch.setattr(model, 'forward_features', features)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = {'image': torch.randn(2, 3, 8, 8), 'label': torch.tensor([0, 1])}
    result = runtime._check_validation(model, batch, policy='full', precision='fp32', device=torch.device('cpu'))
    assert observed == [(False, False, True)]
    assert result['status'] == 'passed' and result['mode'] == 'eval'
    assert result['backward'] == 'not_run' and result['gradients']['status'] == 'not_run'
    assert result['optimizer_updates'] == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())


@pytest.fixture
def mocked_cuda_candidate(monkeypatch):
    """Keep candidate precision routing real; device work is not GPU evidence."""
    runtime = api()
    model = TinyAdapter()
    monkeypatch.setattr(model, 'to', lambda device: model)
    monkeypatch.setattr(runtime, 'build_verified_pretrained_model', lambda *a, **kw: (model, {'status': 'synthetic_fixture_only'}))
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    def check(model, batch, **kwargs):
        return {'status': 'passed', 'policy': kwargs['policy'], 'precision': kwargs['precision']}
    monkeypatch.setattr(runtime, '_check_batch', check)
    monkeypatch.setattr(runtime, '_check_validation', check)
    monkeypatch.setattr(runtime, '_throughput', lambda *a: {'status': 'mocked_not_evidence'})
    return runtime


@pytest.mark.parametrize('native_supported, expected_precision', [(False, 'float16'), (True, 'bfloat16')])
def test_auto_amp_uses_native_bfloat16_capability(mocked_cuda_candidate, tmp_path, monkeypatch, native_supported, expected_precision):
    # Tensor creation/emulation is available in both cases. Only native BF16
    # kernels justify choosing BF16 for the GPU model's auto precision.
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: including_emulation or native_supported)
    config = mocked_cuda_candidate.PreflightConfig(tmp_path / 'prepared', tmp_path / 'output')
    result = mocked_cuda_candidate._run_candidate(config, 'convnext_tiny', {'train': {}, 'val': {}})
    assert result['status'] == 'passed'
    assert len(result['checks']) == 12
    assert {check['precision'] for check in result['checks']} == {'fp32', expected_precision}
    assert sum(check['precision'] == expected_precision for check in result['checks']) == 6


def test_explicit_bfloat16_emulation_is_unsupported(mocked_cuda_candidate, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: including_emulation)
    config = mocked_cuda_candidate.PreflightConfig(tmp_path / 'prepared', tmp_path / 'output', amp='bfloat16')
    result = mocked_cuda_candidate._run_candidate(config, 'convnext_tiny', {'train': {}, 'val': {}})
    assert result['status'] == 'incomplete'
    assert result['checks'][0]['status'] == 'unsupported'
    assert result['checks'][0]['precision'] == 'bfloat16'
    assert all(check['precision'] == 'fp32' for check in result['checks'][1:])


@pytest.mark.parametrize('native_supported', [False, True])
def test_runtime_distinguishes_bfloat16_capabilities(monkeypatch, native_supported):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(
        name='synthetic CUDA fixture', total_memory=16 * 1024**3, major=8 if native_supported else 7, minor=0 if native_supported else 5))
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda including_emulation=True: including_emulation or native_supported)
    cuda = api().collect_runtime()['cuda']
    assert cuda['bfloat16_supported'] is True
    assert cuda['bfloat16_native_supported'] is native_supported
