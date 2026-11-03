"""Offline, weights-only components and the original equal-probability ensemble.

Models remain FP32 for direct feature/Grad-CAM use. Only component_logits applies
CUDA AMP, fixed microbatches, and the Base recipe's CPU FP32 affine head.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle

import torch
from torch import nn
from torch.nn import functional as F

from src.models.pretrained import (
    FB22K_WEIGHT_ID, FB22K_WEIGHT_INFO, V2BASE_WEIGHT_INFO,
    ModelConfig, PretrainedClassifier, create_pretrained_model, load_safe_weight_state,
)
from src.training.checkpoint import _load_model_state, load_model_checkpoint, read_checkpoint
from src.training.data import PreprocessingConfig
from src.training.weights import verify_backbone_state


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _identity(payload):
    names = payload.get('class_names')
    if (not isinstance(names, list) or len(names) != 200
            or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != 200
            or not _digest(payload.get('manifest_sha256'))):
        raise ValueError('Inference component requires 200 unique class names and manifest SHA256')


def _preprocessing(value):
    try:
        if not isinstance(value, dict) or set(value) != {'image_size', 'normalization'}:
            raise ValueError('Preprocessing requires image_size and normalization')
        norm = value['normalization']
        if norm is not None:
            if not isinstance(norm, dict) or set(norm) != {'mean', 'std'}:
                raise ValueError('Invalid preprocessing normalization')
            norm = (norm['mean'], norm['std'])
        return PreprocessingConfig(image_size=value['image_size'], normalization=norm)
    except (TypeError, KeyError) as error:
        raise ValueError('Invalid inference preprocessing') from error


def _describe_preprocessing(config):
    return {'image_size': config.image_size, 'normalization': None if config.normalization is None else
            {'mean': list(config.normalization[0]), 'std': list(config.normalization[1])}}


def _verify_sha256(path, expected_sha256=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        sidecar = path.with_suffix(path.suffix + '.sha256').read_text(encoding='ascii').strip()
    except (OSError, UnicodeError) as error:
        raise ValueError('Inference SHA256 sidecar missing or unreadable') from error
    if (not _digest(sidecar) or sidecar != _sha256(path)
            or expected_sha256 is not None and (not _digest(expected_sha256) or expected_sha256 != sidecar)):
        raise ValueError('Inference SHA256 integrity mismatch')


def _safe_read(path, expected_sha256=None):
    path = Path(path)
    _verify_sha256(path, expected_sha256)
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError, IndexError) as error:
        raise ValueError('Cannot read safe inference payload') from error


def _finite_state(state):
    if not isinstance(state, dict) or not state:
        raise ValueError('Inference model state must be a nonempty tensor mapping')
    for name, value in state.items():
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all())):
            raise ValueError(f'Inference model tensor must be finite float32: {name}')


def load_component(path, *, device='cpu', expected_sha256=None):
    """Verify integrity and restore a self-contained component without downloads."""
    payload = _safe_read(path, expected_sha256)
    required = {'schema_version', 'model_state', 'model_config', 'class_names',
                'manifest_sha256', 'preprocessing', 'provenance'}
    if (not isinstance(payload, dict) or set(payload) != required
            or type(payload['schema_version']) is not int or payload['schema_version'] != 1
            or not isinstance(payload['provenance'], dict)):
        raise ValueError('Unsupported inference component schema')
    _identity(payload)
    _preprocessing(payload['preprocessing'])
    _finite_state(payload['model_state'])
    try:
        config = ModelConfig(**payload['model_config'])
    except (TypeError, KeyError) as error:
        raise ValueError('Invalid inference model config') from error
    model = create_pretrained_model(config, load_weights=False)
    _load_model_state(model, payload['model_state'])
    model.to(torch.device(device)).eval()
    return model, payload


def _settings(value, device):
    required = {'cuda_precision', 'cpu_precision', 'microbatch_size', 'head_device'}
    if (not isinstance(value, dict) or set(value) != required
            or value['cuda_precision'] not in ('float16', 'float32')
            or value['cpu_precision'] != 'float32' or value['head_device'] not in ('cpu', 'model')
            or type(value['microbatch_size']) is not int or value['microbatch_size'] < 1):
        raise ValueError('Invalid inference component settings')
    return {**value, 'resolved_precision': ('cuda_' + value['cuda_precision'] if device.type == 'cuda'
                                           else 'cpu_float32')}


@dataclass
class LoadedClassifier:
    models: dict[str, nn.Module]
    preprocessing: dict[str, PreprocessingConfig]
    class_names: list[str]
    manifest_sha256: str
    device: torch.device
    component_settings: dict

    @torch.inference_mode()
    def component_logits(self, name, images):
        """Return CPU FP32 raw scores using the component's original inference recipe."""
        if name not in self.models:
            raise ValueError(f'Unknown classifier component: {name}')
        size = self.preprocessing[name].image_size
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] < 1
                or tuple(images.shape[1:]) != (3, size, size) or not images.is_floating_point()
                or not torch.isfinite(images).all()):
            raise ValueError(f'{name} images must be finite floating NCHW with shape (N,3,{size},{size})')
        model, settings = self.models[name], self.component_settings[name]
        model.eval()
        cpu_head = settings['head_device'] == 'cpu'
        if cpu_head:
            if not isinstance(model.head, nn.Sequential) or not isinstance(model.head[-1], nn.Linear):
                raise ValueError('CPU affine inference requires a linear component head')
            weight = model.head[-1].weight.detach().to(device='cpu', dtype=torch.float32)
            bias = model.head[-1].bias
            bias = None if bias is None else bias.detach().to(device='cpu', dtype=torch.float32)
        outputs = []
        for chunk in images.split(settings['microbatch_size']):
            chunk = chunk.to(device=self.device, dtype=torch.float32)
            amp = self.device.type == 'cuda' and settings['cuda_precision'] == 'float16'
            context = torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext()
            with context:
                scores = model.forward_features(chunk)['pooled'] if cpu_head else model(chunk)
            scores = scores.to(device='cpu', dtype=torch.float32)
            outputs.append(scores)
        scores = torch.cat(outputs)
        if cpu_head:
            # Preserve the original CPU GEMM shape: one affine call per outer batch.
            scores = F.linear(scores, weight, bias)
        if scores.shape != (len(images), len(self.class_names)) or not torch.isfinite(scores).all():
            raise ValueError(f'Invalid component logits: {name}')
        return scores

    @torch.inference_mode()
    def predict_probabilities(self, batch_by_name):
        """Average each component's softmax probabilities equally, on CPU FP32."""
        if not isinstance(batch_by_name, dict) or set(batch_by_name) != set(self.models):
            raise ValueError('Prediction requires exactly one batch per classifier component')
        counts = [len(images) if isinstance(images, torch.Tensor) and images.ndim else -1
                  for images in batch_by_name.values()]
        if len(set(counts)) != 1:
            raise ValueError('Component batches must contain the same ordered samples')
        probabilities = [self.component_logits(name, batch_by_name[name]).softmax(1) for name in self.models]
        return torch.stack(probabilities).mean(0)


def load_classifier(bundle_json, *, device='cpu'):
    """Load relative components, requiring agreement on class and preparation identity."""
    path, device = Path(bundle_json), torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Classifier device must be cpu or cuda')
    _verify_sha256(path)
    try:
        bundle = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError('Cannot read classifier bundle JSON') from error
    required = {'schema_version', 'models', 'class_names', 'manifest_sha256', 'preprocessing', 'aggregation'}
    if (not isinstance(bundle, dict) or set(bundle) != required
            or type(bundle['schema_version']) is not int or bundle['schema_version'] != 1
            or bundle['aggregation'] != 'equal_probability_mean'
            or not isinstance(bundle['models'], dict) or not bundle['models']
            or not isinstance(bundle['preprocessing'], dict)
            or set(bundle['models']) != set(bundle['preprocessing'])):
        raise ValueError('Unsupported classifier bundle schema')
    _identity(bundle)
    models, preprocessing, settings = {}, {}, {}
    for name, entry in bundle['models'].items():
        if (not isinstance(name, str) or not name or not isinstance(entry, dict)
                or set(entry) != {'path', 'sha256', 'settings'} or not isinstance(entry['path'], str)):
            raise ValueError('Invalid classifier component entry')
        relative = Path(entry['path'])
        resolved = (path.parent / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(path.parent.resolve()):
            raise ValueError('Classifier component path must stay relative to its bundle')
        if not _digest(entry['sha256']):
            raise ValueError('Classifier component requires full SHA256')
        settings[name] = _settings(entry['settings'], device)
        preprocessing[name] = _preprocessing(bundle['preprocessing'][name])
        model, payload = load_component(resolved, device=device, expected_sha256=entry['sha256'])
        if any(payload[key] != bundle[key] for key in ('class_names', 'manifest_sha256')):
            raise ValueError('Classifier component identity mismatch')
        if _preprocessing(payload['preprocessing']) != preprocessing[name]:
            raise ValueError('Classifier component preprocessing mismatch')
        models[name] = model
    return LoadedClassifier(models, preprocessing, list(bundle['class_names']),
                            bundle['manifest_sha256'], device, settings)


def export_classifier_bundle(tiny_checkpoint, base_weights, base_head, output_dir):
    """Export the selected Tiny + Base recipe without changing source checkpoints.

    base_head.folded_raw_head already includes its train-only score scale. Its
    coefficients are copied verbatim: alpha is provenance, never applied again.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f'Classifier output directory already exists: {output}')
    tiny_checkpoint, base_weights, base_head = map(Path, (tiny_checkpoint, base_weights, base_head))
    tiny_payload = read_checkpoint(tiny_checkpoint)
    identity = {'class_names': tiny_payload['class_names'],
                'manifest_sha256': tiny_payload['data_identity']['manifest_sha256']}
    _identity(identity)
    tiny_config = ModelConfig(**tiny_payload['model_config'])
    provenance = tiny_payload['weight_provenance']
    if (tiny_config.architecture != 'convnext_tiny' or tiny_config.weights != FB22K_WEIGHT_ID
            or provenance.get('weight_id') != FB22K_WEIGHT_ID
            or provenance.get('sha256') != FB22K_WEIGHT_INFO['hash_prefix']):
        raise ValueError('Tiny checkpoint must use the pinned pretrained 22K weight profile')
    train = tiny_payload['training_config']
    tiny_preprocessing = PreprocessingConfig(image_size=train['image_size'], normalization=train['normalization'])
    head = _safe_read(base_head)
    base_digest = _sha256(base_weights)
    if base_digest != V2BASE_WEIGHT_INFO['hash_prefix']:
        raise ValueError('Base pretrained weights SHA256 mismatch')
    if (not isinstance(head, dict) or head.get('profile') != V2BASE_WEIGHT_INFO['source_profile']
            or head.get('weight_sha256') != base_digest
            or any(head.get(key) != value for key, value in identity.items())):
        raise ValueError('Base head profile, pretrained weights or class/manifest identity mismatch')
    scale = head.get('score_scale')
    if (not isinstance(scale, dict) or scale.get('fit_partition') != 'train'
            or scale.get('validation_used_for_fitting') is not False
            or type(scale.get('alpha')) not in (int, float)
            or not 0 < scale['alpha'] < float('inf')):
        raise ValueError('Base head score_scale must be finite, positive and fitted on train only')
    folded = head.get('folded_raw_head')
    shapes = {'weight': (200, 1024), 'bias': (200,)}
    if not isinstance(folded, dict) or set(folded) != set(shapes):
        raise ValueError('Base folded_raw_head requires exactly weight and bias')
    for key, shape in shapes.items():
        tensor = folded[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()):
            raise ValueError(f'Base head tensor must be finite float32 with shape {shape}: {key}')
    tiny_model, _ = load_model_checkpoint(tiny_checkpoint, device='cpu')
    import timm
    native_state = load_safe_weight_state(base_weights)
    _finite_state(native_state)
    native = timm.create_model(V2BASE_WEIGHT_INFO['source_profile'], pretrained=False)
    _load_model_state(native, native_state)
    base_model = PretrainedClassifier(ModelConfig('convnextv2_base'), native, weights_loaded=True)
    verify_backbone_state(base_model, native_state)
    base_model.head[1].load_state_dict(folded, strict=True)
    base_model.eval()
    components = {
        'tiny': (tiny_model, tiny_preprocessing,
                 {'source': 'fine_tuned_checkpoint', 'source_checkpoint_sha256': _sha256(tiny_checkpoint),
                  'pretrained_weight_id': tiny_config.weights,
                  'pretrained_weight_sha256': provenance['sha256']},
                 {'cuda_precision': 'float16', 'cpu_precision': 'float32',
                  'microbatch_size': 8, 'head_device': 'model'}),
        'base': (base_model, PreprocessingConfig(image_size=384),
                 {'source': 'pretrained_encoder_with_train_scaled_ridge_head',
                  'pretrained_weight_id': base_model.config.weights, 'pretrained_weight_sha256': base_digest,
                  'source_head_sha256': _sha256(base_head), 'score_scale_alpha_already_folded': scale['alpha'],
                  'score_scale_fit_partition': 'train', 'validation_used_for_score_scaling': False},
                 {'cuda_precision': 'float32', 'cpu_precision': 'float32',
                  'microbatch_size': 4, 'head_device': 'cpu'}),
    }
    bundle = {'schema_version': 1, **identity, 'models': {}, 'preprocessing': {},
              'aggregation': 'equal_probability_mean'}
    # Validate before creating a new output directory; never overwrite an export.
    for model, _, _, _ in components.values():
        _finite_state(model.state_dict())
    output.mkdir(parents=True, exist_ok=False)
    for name, (model, preprocessing, proof, settings) in components.items():
        recipe = _describe_preprocessing(preprocessing)
        payload = {'schema_version': 1, 'model_config': asdict(model.config),
                   'model_state': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                   **identity, 'preprocessing': recipe, 'provenance': proof}
        artifact = output / f'{name}.pt'
        torch.save(payload, artifact)
        digest = _sha256(artifact)
        artifact.with_suffix('.pt.sha256').write_text(digest + '\n', encoding='ascii')
        bundle['models'][name] = {'path': artifact.name, 'sha256': digest, 'settings': settings}
        bundle['preprocessing'][name] = recipe
    manifest = output / 'bundle.json'
    manifest.write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    manifest.with_suffix('.json.sha256').write_text(_sha256(manifest) + '\n', encoding='ascii')
    return manifest
