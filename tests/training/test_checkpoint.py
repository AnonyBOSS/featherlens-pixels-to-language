import copy
from dataclasses import asdict
import hashlib
import importlib
import random
import numpy as np
import pytest
import torch
from test_trainer import DATA_IDENTITY, PROVENANCE, TinyAdapter, tiny_loaders
from collections import OrderedDict
import importlib.util
import io
import json
from pathlib import Path
from torchvision import models
from src.models.pretrained import ModelConfig, PretrainedClassifier


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


ARCHITECTURES = ("convnext_tiny", "efficientnet_b3", "regnet_y_3_2gf")
EXPECTED = {
    "convnext_tiny": ("ConvNeXt_Tiny_Weights.IMAGENET1K_V1", "convnext_tiny-983f1562.pth", "983f1562"),
    "efficientnet_b3": ("EfficientNet_B3_Weights.IMAGENET1K_V1", "efficientnet_b3_rwightman-b3899882.pth", "b3899882"),
    "regnet_y_3_2gf": ("RegNet_Y_3_2GF_Weights.IMAGENET1K_V2", "regnet_y_3_2gf-9180c971.pth", "9180c971"),
}


@pytest.fixture(params=ARCHITECTURES, scope="module")
def synthetic(request):
    """Real adapter and independent random native state; no pretrained claim."""
    architecture = request.param
    native = getattr(models, architecture)(weights=None)
    state = OrderedDict((key, value.detach().clone()) for key, value in native.state_dict().items())
    adapter = PretrainedClassifier(ModelConfig(architecture), native, weights_loaded=False)
    return adapter, state


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_resolve_exact_pinned_official_identity_without_io(api, architecture, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("metadata must not access cache"))
    info = api.resolve_weight_info(ModelConfig(architecture))
    weight_id, filename, prefix = EXPECTED[architecture]
    assert info["weight_id"] == weight_id
    assert info["url"] == "https://download.pytorch.org/models/" + filename
    assert info["filename"] == filename
    assert info["hash_prefix"] == prefix
    assert info["architecture"] == architecture
    assert info["torch_version"] == torch.__version__
    import torchvision
    assert info["torchvision_version"] == torchvision.__version__
    json.dumps(info)


def test_invalid_configuration_and_switches_fail_before_io(api, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("invalid arguments accessed cache"))
    with pytest.raises(TypeError, match="ModelConfig"):
        api.resolve_weight_info({"architecture": "convnext_tiny"})
    for kwargs in ({"offline": 1}, {"progress": 1}):
        with pytest.raises(TypeError, match="bool"):
            api.prepare_official_weight(ModelConfig("convnext_tiny"), **kwargs)


def test_missing_offline_checkpoint_does_not_download_or_make_cache(api, tmp_path, monkeypatch):
    cache = tmp_path / "not-created"
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache))
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("offline network access"))
    with pytest.raises(FileNotFoundError, match="offline"):
        api.prepare_official_weight(ModelConfig("convnext_tiny"), offline=True)
    assert not cache.exists()


def _synthetic_cache(monkeypatch, tmp_path, payload):
    """Replace only the external URL/cache seams with a hash-identifiable fixture."""
    digest = hashlib.sha256(payload).hexdigest()
    filename = f"synthetic-{digest[:8]}.pth"
    weight = models.EfficientNet_B3_Weights.IMAGENET1K_V1
    monkeypatch.setattr(weight.value, "url", "https://download.pytorch.org/models/" + filename)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    return tmp_path / "checkpoints" / filename, digest


def test_existing_valid_cache_records_full_digest_without_download(api, tmp_path, monkeypatch):
    payload = b"synthetic cache bytes, not pretrained weights"
    path, digest = _synthetic_cache(monkeypatch, tmp_path, payload)
    path.parent.mkdir()
    path.write_bytes(payload)
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("duplicate download"))
    info = api.prepare_official_weight(ModelConfig("efficientnet_b3"), offline=True)
    assert info["cache_path"] == str(path)
    assert info["sha256"] == digest
    assert info["size_bytes"] == len(payload)
    assert info["hash_verified"] is True
    json.dumps(info)


def test_supported_download_uses_hash_then_reuses_one_artifact(api, tmp_path, monkeypatch):
    payload = b"synthetic downloaded bytes, not official evidence"
    path, digest = _synthetic_cache(monkeypatch, tmp_path, payload)
    calls = []
    def download(url, destination, *, hash_prefix, progress):
        assert url == "https://download.pytorch.org/models/" + path.name
        assert hash_prefix == digest[:8]
        assert progress is False
        calls.append(destination)
        Path(destination).write_bytes(payload)
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    first = api.prepare_official_weight(ModelConfig("efficientnet_b3"), progress=False)
    second = api.prepare_official_weight(ModelConfig("efficientnet_b3"), offline=True)
    assert first == second
    assert len(calls) == 1
    assert first["sha256"] == digest


@pytest.mark.parametrize("cached", [True, False])
def test_corrupt_bytes_fail_hash_verification_without_fallback(api, tmp_path, monkeypatch, cached):
    path, _ = _synthetic_cache(monkeypatch, tmp_path, b"expected synthetic bytes")
    if cached:
        path.parent.mkdir()
        path.write_bytes(b"corrupt")
    def download(url, destination, **kwargs):
        if cached:
            pytest.fail("corrupt cache must not silently redownload")
        Path(destination).write_bytes(b"corrupt")
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    with pytest.raises(ValueError, match="SHA-256"):
        api.prepare_official_weight(ModelConfig("efficientnet_b3"))


def test_download_failure_propagates(api, tmp_path, monkeypatch):
    _synthetic_cache(monkeypatch, tmp_path, b"synthetic")
    def unavailable(*args, **kwargs):
        raise OSError("deliberate unavailable download")
    monkeypatch.setattr(torch.hub, "download_url_to_file", unavailable)
    with pytest.raises(OSError, match="unavailable"):
        api.prepare_official_weight(ModelConfig("efficientnet_b3"))


def test_all_native_retained_tensors_match_exactly(api, synthetic):
    model, state = synthetic
    proof = api.verify_backbone_state(model, state)
    retained = {key: value for key, value in model.state_dict().items() if not key.startswith("head.")}
    assert proof["status"] == "verified"
    assert proof["retained_tensor_count"] == len(retained)
    assert proof["parameter_tensor_count"] == sum(not key.startswith("head.") for key, _ in model.named_parameters())
    assert proof["buffer_tensor_count"] == len(list(model.named_buffers()))
    assert len(proof["excluded_native_head_keys"]) == 2
    json.dumps(proof)


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "value", "type"])
def test_incompatible_retained_state_is_rejected(api, synthetic, corruption):
    model, native_state = synthetic
    state = OrderedDict(native_state)
    key = next(iter(state))
    if corruption == "missing":
        state.pop(key)
    elif corruption == "extra":
        state["unexpected.weight"] = torch.ones(1)
    elif corruption == "shape":
        state[key] = state[key].reshape(-1)[:1]
    elif corruption == "dtype":
        state[key] = state[key].double()
    elif corruption == "value":
        state[key] = state[key].clone()
        state[key].view(-1)[0] += 1
    else:
        state[key] = [1]
    with pytest.raises((TypeError, ValueError), match="missing|unexpected|shape|dtype|value|tensor"):
        api.verify_backbone_state(model, state)


def test_buffers_and_convnext_normalization_are_verified(api, synthetic):
    model, native_state = synthetic
    target = "classifier.0.weight" if model.config.architecture == "convnext_tiny" else next(
        key for key in native_state if key.endswith("running_mean")
    )
    state = OrderedDict(native_state)
    state[target] = state[target].clone()
    state[target].view(-1)[0] += 1
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, state)


def test_only_the_expected_native_1000_class_head_is_excluded(api, synthetic):
    model, native_state = synthetic
    head_weight = {"convnext_tiny": "classifier.2.weight", "efficientnet_b3": "classifier.1.weight", "regnet_y_3_2gf": "fc.weight"}[model.config.architecture]
    state = OrderedDict(native_state)
    state[head_weight] = torch.zeros(200, native_state[head_weight].shape[1])
    with pytest.raises(ValueError, match="1000|shape"):
        api.verify_backbone_state(model, state)
    state = OrderedDict(native_state)
    state.pop(head_weight)
    with pytest.raises(ValueError, match="missing"):
        api.verify_backbone_state(model, state)


def test_new_head_must_have_200_trainable_outputs(api, synthetic):
    model, state = synthetic
    model.head[1].weight.requires_grad_(False)
    try:
        with pytest.raises(ValueError, match="trainable"):
            api.verify_backbone_state(model, state)
    finally:
        model.head[1].weight.requires_grad_(True)
    original = model.head[1]
    model.head[1] = torch.nn.Linear(original.in_features, 201)
    try:
        with pytest.raises(ValueError, match="200"):
            api.verify_backbone_state(model, state)
    finally:
        model.head[1] = original


def test_verified_builder_uses_real_factory_and_safe_cpu_checkpoint(api, tmp_path, monkeypatch):
    native = models.efficientnet_b3(weights=None)
    state = native.state_dict()
    stream = io.BytesIO()
    torch.save(state, stream)
    path, digest = _synthetic_cache(monkeypatch, tmp_path, stream.getvalue())
    path.parent.mkdir()
    path.write_bytes(stream.getvalue())
    del native, stream
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("unexpected network"))
    real_load = torch.load
    loads = []
    def safe_load(*args, **kwargs):
        loads.append(kwargs)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", safe_load)
    model, proof = api.build_verified_pretrained_model(ModelConfig("efficientnet_b3"), offline=True, progress=False)
    assert model.weights_loaded is True
    assert model.loaded_weight_id == "EfficientNet_B3_Weights.IMAGENET1K_V1"
    assert proof["sha256"] == digest
    assert proof["backbone"]["status"] == "verified"
    assert proof["head"] == {"out_features": 200, "trainable": True}
    assert any(item.get("weights_only") is True and item.get("map_location") == "cpu" for item in loads)
    json.dumps(proof)


@pytest.mark.parametrize("payload_kind", ["not_mapping", "unsafe_pickle", "wrong_state"])
def test_builder_rejects_invalid_checkpoint_without_fallback(api, tmp_path, monkeypatch, payload_kind):
    stream = io.BytesIO()
    payload = [torch.zeros(1)] if payload_kind == "not_mapping" else (
        Path("unsafe-global") if payload_kind == "unsafe_pickle" else {"bad": torch.zeros(1)}
    )
    torch.save(payload, stream)
    path, _ = _synthetic_cache(monkeypatch, tmp_path, stream.getvalue())
    path.parent.mkdir()
    path.write_bytes(stream.getvalue())
    if payload_kind != "wrong_state":
        monkeypatch.setattr(api, "create_pretrained_model", lambda *a, **k: pytest.fail("invalid checkpoint reached factory"))
    with pytest.raises((TypeError, ValueError, RuntimeError), match="checkpoint|state|missing|load|tensor"):
        api.build_verified_pretrained_model(ModelConfig("efficientnet_b3"), offline=True)


def test_even_one_ulp_retained_value_difference_is_rejected(api, synthetic):
    model, native_state = synthetic
    state = OrderedDict(native_state)
    key = next(iter(state))
    state[key] = state[key].clone()
    scalar = state[key].view(-1)
    scalar[0] = torch.nextafter(scalar[0], torch.full_like(scalar[0], float("inf")))
    assert torch.allclose(state[key], native_state[key])
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, state)


@pytest.mark.parametrize("synthetic", ["efficientnet_b3", "regnet_y_3_2gf"], indirect=True)
def test_integer_batch_counter_is_verified(api, synthetic):
    model, native_state = synthetic
    key = next(key for key in native_state if key.endswith("num_batches_tracked"))
    state = OrderedDict(native_state)
    state[key] = state[key] + 1
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, state)


def test_builder_rejects_factory_state_disagreement(api, tmp_path, monkeypatch):
    native = models.efficientnet_b3(weights=None)
    state = native.state_dict()
    stream = io.BytesIO()
    torch.save(state, stream)
    path, _ = _synthetic_cache(monkeypatch, tmp_path, stream.getvalue())
    path.parent.mkdir()
    path.write_bytes(stream.getvalue())
    key = next(key for key in state if key.endswith("running_mean"))
    state[key] = state[key] + 1
    def mismatched_download(*, progress, check_hash):
        return state
    monkeypatch.setattr(models.EfficientNet_B3_Weights.IMAGENET1K_V1, "get_state_dict", mismatched_download)
    with pytest.raises(ValueError, match="value"):
        api.build_verified_pretrained_model(ModelConfig("efficientnet_b3"), offline=True)


def test_unsupported_filename_hash_fails_before_cache_io(api, monkeypatch):
    monkeypatch.setattr(models.EfficientNet_B3_Weights.IMAGENET1K_V1.value, "url", "https://download.pytorch.org/models/invalid.pth")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("unsupported artifact accessed cache"))
    with pytest.raises(ValueError, match="hash prefix"):
        api.prepare_official_weight(ModelConfig("efficientnet_b3"), offline=True)


def test_fb_profile_resolves_full_hash_and_immutable_revision_without_io(api, monkeypatch):
    from src.models.pretrained import FB22K_WEIGHT_ID
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("metadata accessed cache"))
    info = api.resolve_weight_info(ModelConfig("convnext_tiny", weights=FB22K_WEIGHT_ID))
    assert info["weight_id"] == FB22K_WEIGHT_ID
    assert info["revision"] == "32c741ba106f9e062174ef829354a16719c3dcac"
    assert info["hash_prefix"] == "c1253ad2d4a751cb0b1fb92a850f99e8f426a701d6e7600b03a490557d77bdcd"
    assert "/resolve/" + info["revision"] + "/" in info["url"]
    assert info["hash_prefix"] in info["filename"]


def test_v2base_pinned_metadata_and_safe_cpu_state_reader(api, tmp_path, monkeypatch):
    from src.models.pretrained import V2BASE_WEIGHT_ID, load_safe_weight_state
    save_file = pytest.importorskip("safetensors.torch").save_file
    monkeypatch.setattr(torch.hub, "get_dir", lambda: pytest.fail("metadata accessed cache"))
    info = api.resolve_weight_info(ModelConfig("convnextv2_base"))
    assert info["weight_id"] == V2BASE_WEIGHT_ID
    assert info["revision"] == "bea9dc139d5981aff6ce4d6d9690086e45ad8b56"
    assert info["hash_prefix"] == "972bcbb8d9965eca7a2f72d9592f54dfc890f9e6aef920cea4269963f46f2f42"
    assert info["source_profile"] == "convnextv2_base.fcmae_ft_in22k_in1k_384"
    path = tmp_path / "small.safetensors"
    state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    save_file(state, str(path))
    loaded = load_safe_weight_state(path)
    assert loaded["weight"].device.type == "cpu"
    assert torch.equal(loaded["weight"], state["weight"])


def test_v2base_proof_checks_all_retained_tensors_and_native_head(api):
    timm = pytest.importorskip("timm")
    native = timm.create_model("convnextv2_base", pretrained=False)
    state = native.state_dict()
    model = PretrainedClassifier(ModelConfig("convnextv2_base"), native, weights_loaded=False)
    proof = api.verify_backbone_state(model, state)
    assert proof["retained_tensor_count"] == 378
    assert proof["excluded_native_head_keys"] == ["head.fc.bias", "head.fc.weight"]
    bad = dict(state)
    bad["head.norm.bias"] = bad["head.norm.bias"] + 1
    with pytest.raises(ValueError, match="value"):
        api.verify_backbone_state(model, bad)
    bad = dict(state)
    bad["head.fc.weight"] = bad["head.fc.weight"][:200]
    with pytest.raises(ValueError, match="1000|shape"):
        api.verify_backbone_state(model, bad)
