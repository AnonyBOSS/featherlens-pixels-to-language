"""Portable inference contracts, using tiny real torch modules instead of CNN allocations."""
from dataclasses import asdict
import hashlib
import importlib
import json

import pytest
import torch
from torch import nn

from src.models.pretrained import ModelConfig

NAMES = [f'bird {index}' for index in range(200)]
MANIFEST = 'a' * 64
PREPROCESSING = {'image_size': 4, 'normalization': None}
SETTINGS = {'cuda_precision': 'float32', 'cpu_precision': 'float32',
            'microbatch_size': 2, 'head_device': 'model'}


def api():
    try:
        return importlib.import_module('src.evaluation.inference')
    except ModuleNotFoundError as error:
        pytest.fail(f'Portable inference API is missing: {error}')


class SmallClassifier(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig('convnext_tiny')
        self.head = nn.Sequential(nn.Identity(), nn.Linear(3, 200))
        self.batch_sizes = []
        with torch.no_grad():
            self.head[1].weight.zero_()
            self.head[1].bias.zero_()
            self.head[1].weight[0, 0] = 2
            self.head[1].weight[1, 1] = -1

    def forward_features(self, images):
        self.batch_sizes.append(len(images))
        return {'pooled': images.mean((2, 3)), 'spatial': images}

    def forward(self, images):
        return self.head(self.forward_features(images)['pooled'])


def write_component(path, *, bias=0, preprocessing=None, manifest=MANIFEST):
    model = SmallClassifier()
    with torch.no_grad():
        model.head[1].bias[0] = bias
    payload = {'schema_version': 1, 'model_state': model.state_dict(),
               'model_config': asdict(model.config), 'class_names': list(NAMES),
               'manifest_sha256': manifest,
               'preprocessing': preprocessing or PREPROCESSING,
               'provenance': {'source_sha256': 'b' * 64}}
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.pt.sha256').write_text(digest + '\n')
    return payload, digest


def write_bundle(tmp_path, components):
    bundle = {'schema_version': 1, 'models': {}, 'class_names': NAMES,
              'manifest_sha256': MANIFEST, 'preprocessing': {},
              'aggregation': 'equal_probability_mean'}
    for name, bias in components.items():
        path = tmp_path / f'{name}.pt'
        _, digest = write_component(path, bias=bias)
        bundle['models'][name] = {'path': path.name, 'sha256': digest, 'settings': SETTINGS}
        bundle['preprocessing'][name] = PREPROCESSING
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(bundle))
    path.with_suffix('.json.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    return path, bundle


def patch_small_model(monkeypatch):
    def create(config, *, load_weights):
        assert load_weights is False  # Reload must remain offline.
        return SmallClassifier(config)
    monkeypatch.setattr(api(), 'create_pretrained_model', create)


def test_equal_probability_mean_preserves_component_scores_and_microbatches(tmp_path, monkeypatch):
    module = api()
    patch_small_model(monkeypatch)
    path, _ = write_bundle(tmp_path, {'tiny': 0, 'base': 1})
    loaded = module.load_classifier(path)
    images = torch.ones(5, 3, 4, 4, requires_grad=True)
    logits = loaded.component_logits('tiny', images)
    assert logits.shape == (5, 200) and logits.device.type == 'cpu'
    assert logits.dtype == torch.float32 and not logits.requires_grad
    assert logits[0, :3].tolist() == [2, -1, 0]
    assert loaded.models['tiny'].batch_sizes == [2, 2, 1]
    probabilities = loaded.predict_probabilities({'tiny': images, 'base': images})
    # Hand derived: tiny class 0 exp(2), class 1 exp(-1), 198 scores zero;
    # base differs only by its already-loaded +1 class-0 affine bias.
    expected = 0.5 * (torch.exp(torch.tensor(2.)) / (torch.exp(torch.tensor(2.)) + torch.exp(torch.tensor(-1.)) + 198)
                      + torch.exp(torch.tensor(3.)) / (torch.exp(torch.tensor(3.)) + torch.exp(torch.tensor(-1.)) + 198))
    assert torch.allclose(probabilities[:, 0], expected.expand(5))
    assert torch.allclose(probabilities.sum(1), torch.ones(5))
    assert loaded.component_settings['tiny']['resolved_precision'] == 'cpu_float32'
    assert images.grad is None


def test_cpu_affine_head_uses_pooled_scores_without_second_scale(tmp_path, monkeypatch):
    module = api()
    patch_small_model(monkeypatch)
    path, bundle = write_bundle(tmp_path, {'base': 1})
    bundle['models']['base']['settings'] = {**SETTINGS, 'head_device': 'cpu'}
    path.write_text(json.dumps(bundle))
    path.with_suffix('.json.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = module.load_classifier(path)
    loaded.models['base'].forward = lambda images: pytest.fail('must use the pooled encoder and CPU head')
    affine_batches = []
    original_linear = module.F.linear
    def affine(inputs, weight, bias):
        affine_batches.append(tuple(inputs.shape))
        return original_linear(inputs, weight, bias)
    monkeypatch.setattr(module.F, 'linear', affine)
    scores = loaded.component_logits('base', torch.ones(3, 3, 4, 4))
    assert affine_batches == [(3, 3)]  # Original recipe applies its CPU head once per outer batch.
    assert scores[:, :3].tolist() == [[3, -1, 0]] * 3


def test_component_reload_rejects_integrity_and_tensor_damage(tmp_path, monkeypatch):
    module = api()
    patch_small_model(monkeypatch)
    path = tmp_path / 'tiny.pt'
    payload, digest = write_component(path)
    model, restored = module.load_component(path, expected_sha256=digest)
    assert not model.training and restored['class_names'] == NAMES
    with pytest.raises(ValueError, match='SHA256'):
        module.load_component(path, expected_sha256='f' * 64)
    payload['model_state']['head.1.weight'][0, 0] = float('nan')
    torch.save(payload, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='finite'):
        module.load_component(path)
    path.with_suffix('.pt.sha256').unlink()
    with pytest.raises(ValueError, match='SHA256'):
        module.load_component(path)


@pytest.mark.parametrize('damage', ['identity', 'preprocessing', 'escape', 'settings'])
def test_bundle_rejects_inconsistent_or_nonportable_components(tmp_path, monkeypatch, damage):
    module = api()
    patch_small_model(monkeypatch)
    path, bundle = write_bundle(tmp_path, {'tiny': 0})
    if damage == 'identity':
        bundle['manifest_sha256'] = 'c' * 64
    elif damage == 'preprocessing':
        bundle['preprocessing']['tiny'] = {'image_size': 8, 'normalization': None}
    elif damage == 'escape':
        bundle['models']['tiny']['path'] = '../tiny.pt'
    else:
        bundle['models']['tiny']['settings'] = {**SETTINGS, 'microbatch_size': 0}
    path.write_text(json.dumps(bundle))
    path.with_suffix('.json.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        module.load_classifier(path)


def test_predict_rejects_missing_misaligned_and_invalid_images(tmp_path, monkeypatch):
    module = api()
    patch_small_model(monkeypatch)
    path, _ = write_bundle(tmp_path, {'tiny': 0, 'base': 1})
    loaded = module.load_classifier(path)
    images = torch.ones(2, 3, 4, 4)
    for batch in ({'tiny': images}, {'tiny': images, 'base': images[:1]},
                  {'tiny': images, 'base': torch.ones(2, 3, 3, 3)},
                  {'tiny': images, 'base': torch.full_like(images, float('inf'))}):
        with pytest.raises(ValueError):
            loaded.predict_probabilities(batch)

class SmallBaseNative(nn.Module):
    """Keep the real Base adapter/verification while avoiding a 350 MB encoder."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 1024, 1)
        self.stages = nn.Sequential(*[nn.Identity() for _ in range(4)])
        self.norm_pre = nn.Identity()
        self.head = nn.Module()
        self.head.pre_logits = nn.Identity()
        self.head.drop = nn.Dropout(0)
        self.head.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head.norm = nn.Identity()
        self.head.fc = nn.Linear(1024, 1000)


def export_inputs(tmp_path, monkeypatch):
    module = api()
    import timm
    from src.models.pretrained import FB22K_WEIGHT_ID, FB22K_WEIGHT_INFO
    tiny = SmallClassifier(ModelConfig('convnext_tiny', weights=FB22K_WEIGHT_ID))
    checkpoint = {'model_config': asdict(tiny.config), 'model_state': tiny.state_dict(),
                  'class_names': NAMES, 'data_identity': {'manifest_sha256': MANIFEST, 'class_names': NAMES},
                  'training_config': {'image_size': 4, 'normalization': None},
                  'weight_provenance': {'sha256': FB22K_WEIGHT_INFO['hash_prefix'],
                                        'weight_id': FB22K_WEIGHT_ID, 'cache_path': 'C:/private/cache'}}
    tiny_path = tmp_path / 'checkpoint.pt'
    tiny_path.write_bytes(b'Original trainer checkpoint stays immutable')
    monkeypatch.setattr(module, 'read_checkpoint', lambda path: checkpoint)
    monkeypatch.setattr(module, 'load_model_checkpoint', lambda path, **kwargs: (tiny, checkpoint))
    monkeypatch.setattr(timm, 'create_model', lambda profile, *, pretrained: SmallBaseNative() if not pretrained else pytest.fail('offline'))
    native = SmallBaseNative()
    weights = tmp_path / 'base-native.pt'
    torch.save(native.state_dict(), weights)
    weight_digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setitem(module.V2BASE_WEIGHT_INFO, 'hash_prefix', weight_digest)
    head = {'profile': module.V2BASE_WEIGHT_INFO['source_profile'], 'weight_sha256': weight_digest,
            'manifest_sha256': MANIFEST, 'class_names': NAMES,
            'folded_raw_head': {'weight': torch.full((200, 1024), 0.5), 'bias': torch.ones(200)},
            'score_scale': {'alpha': 21.160089690544176, 'fit_partition': 'train',
                            'validation_used_for_fitting': False}, 'source_head_sha256': 'c' * 64}
    head_path = tmp_path / 'ridge-scaled-head.pt'
    torch.save(head, head_path)
    head_path.with_suffix('.pt.sha256').write_text(hashlib.sha256(head_path.read_bytes()).hexdigest())
    return tiny_path, weights, head_path, head


def test_export_is_portable_compact_and_preserves_already_scaled_head(tmp_path, monkeypatch):
    module = api()
    tiny, weights, head, _ = export_inputs(tmp_path, monkeypatch)
    originals = [path.read_bytes() for path in (tiny, weights, head)]
    output = tmp_path / 'delivery'
    manifest_path = module.export_classifier_bundle(tiny, weights, head, output)
    bundle = json.loads(manifest_path.read_text())
    assert manifest_path == output / 'bundle.json'
    assert bundle['class_names'] == NAMES and bundle['manifest_sha256'] == MANIFEST
    assert set(bundle['models']) == {'tiny', 'base'}
    for name, entry in bundle['models'].items():
        artifact = output / entry['path']
        payload = torch.load(artifact, weights_only=True, map_location='cpu')
        assert entry['sha256'] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert artifact.with_suffix('.pt.sha256').read_text().strip() == entry['sha256']
        assert not {'optimizer_state', 'rng_state', 'epochs_completed'} & payload.keys()
        assert 'C:/private' not in json.dumps(payload['provenance'])
        assert payload['preprocessing'] == bundle['preprocessing'][name]
        if name == 'base':
            assert torch.equal(payload['model_state']['head.1.weight'], torch.full((200, 1024), 0.5))
            assert torch.equal(payload['model_state']['head.1.bias'], torch.ones(200))
    assert [path.read_bytes() for path in (tiny, weights, head)] == originals
    with pytest.raises(FileExistsError):
        module.export_classifier_bundle(tiny, weights, head, output)


@pytest.mark.parametrize('damage', ['validation_scaling', 'profile', 'identity', 'head_shape', 'nonfinite', 'weights'])
def test_export_rejects_mismatched_or_unapproved_base_artifact(tmp_path, monkeypatch, damage):
    module = api()
    tiny, weights, head_path, head = export_inputs(tmp_path, monkeypatch)
    if damage == 'validation_scaling':
        head['score_scale']['validation_used_for_fitting'] = True
    elif damage == 'profile':
        head['profile'] = 'different profile'
    elif damage == 'identity':
        head['class_names'] = list(reversed(NAMES))
    elif damage == 'head_shape':
        head['folded_raw_head']['weight'] = torch.ones(1, 1024)
    elif damage == 'nonfinite':
        head['folded_raw_head']['bias'][0] = float('inf')
    else:
        weights.write_bytes(b'damaged weights')
    torch.save(head, head_path)
    head_path.with_suffix('.pt.sha256').write_text(hashlib.sha256(head_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        module.export_classifier_bundle(tiny, weights, head_path, tmp_path / 'delivery')



def test_malformed_class_mapping_is_rejected_as_input_error(tmp_path, monkeypatch):
    module = api()
    patch_small_model(monkeypatch)
    path = tmp_path / 'tiny.pt'
    payload, _ = write_component(path)
    payload['class_names'][0] = ['not a class name']
    torch.save(payload, path)
    path.with_suffix('.pt.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='class'):
        module.load_component(path)



@pytest.mark.parametrize('damage', ['missing_sidecar', 'malformed_sidecar', 'changed_recipe'])
def test_bundle_json_integrity_is_verified_before_model_loading(tmp_path, monkeypatch, damage):
    module = api()
    path, bundle = write_bundle(tmp_path, {'tiny': 0})
    monkeypatch.setattr(module, 'load_component', lambda *a, **k: pytest.fail('model must not load before JSON integrity passes'))
    if damage == 'missing_sidecar':
        path.with_suffix('.json.sha256').unlink()
    elif damage == 'malformed_sidecar':
        path.with_suffix('.json.sha256').write_text('not a SHA256')
    else:
        bundle['models']['tiny']['settings']['cuda_precision'] = 'float16'
        path.write_text(json.dumps(bundle))  # Deliberately leave the original sidecar unchanged.
    with pytest.raises(ValueError, match='SHA256'):
        module.load_classifier(path)
