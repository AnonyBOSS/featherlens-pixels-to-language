"""Weights-only ensemble loading, evaluation metrics, and single-image prediction."""
from contextlib import nullcontext
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.classification import CubClassificationDataset
from src.data.image_io import load_rgb
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.models.pretrained import (
    FB22K_WEIGHT_ID, FB22K_WEIGHT_INFO, V2BASE_WEIGHT_INFO,
    ModelConfig, PretrainedClassifier, create_pretrained_model, load_safe_weight_state,
)
from src.training.checkpoint import (
    _load_model_state, load_model_checkpoint, read_checkpoint, verify_backbone_state,
)
from src.training.data import PreprocessingConfig, validate_preflight_preparation


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
    """Bundle the fine-tuned ConvNeXt-Tiny checkpoint and Ridge-head ConvNeXtV2-Base model into output_dir."""
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


def classification_report(probabilities, labels, class_names, image_ids):
    """Compute top-1/top-5 accuracy, macro/weighted precision/recall/F1, and confusion matrix."""
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2
            or not probabilities.is_floating_point() or not len(probabilities)):
        raise ValueError('Expected nonempty floating-point NxC probabilities')
    names, identifiers = list(class_names), list(image_ids)
    if (not names or any(not isinstance(n,str) or not n for n in names)
            or len(set(names)) != len(names) or probabilities.shape[1] != len(names)):
        raise ValueError('Ordered class names must be unique and match probabilities')
    if (not isinstance(labels,torch.Tensor) or labels.dtype != torch.int64
            or labels.shape != (len(probabilities),)):
        raise ValueError('Labels must be an int64 vector matching the samples')
    if (len(identifiers) != len(labels) or any(not isinstance(i,str) or not i for i in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError('Each sample needs one unique nonempty image_id')
    p, truth = probabilities.detach().cpu(), labels.detach().cpu()
    if not torch.isfinite(p).all():
        raise FloatingPointError('Non-finite prediction probabilities')
    if ((p < 0).any() or (p > 1).any()
            or not torch.allclose(p.sum(1),torch.ones(len(p),dtype=p.dtype),atol=1e-6,rtol=0)):
        raise ValueError('Probabilities must be nonnegative rows summing to one')
    if ((truth < 0).any() or (truth >= len(names)).any()):
        raise ValueError('Label is outside the ordered class mapping')
    prediction = p.argmax(1)
    ranked = p.argsort(dim=1,descending=True,stable=True)[:, :min(5,len(names))]
    confusion = torch.bincount(truth*len(names)+prediction,minlength=len(names)**2).reshape(len(names),len(names))
    per_class=[]
    for i,name in enumerate(names):
        tp=int(confusion[i,i]);support=int(confusion[i].sum());count=int(confusion[:,i].sum())
        per_class.append({'class_id':i,'class_name':name,'support':support,'predicted_count':count,
            'true_positive':tp,'precision':tp/count if count else 0.,'recall':tp/support if support else 0.,
            'f1':2*tp/(support+count) if support+count else 0.})
    n=len(truth)
    predictions={identifier:{'true_label':int(y),'predicted_label':int(pred),'top5':top,
        'confidence':float(p[row,pred])} for row,(identifier,y,pred,top) in
        enumerate(zip(identifiers,truth.tolist(),prediction.tolist(),ranked.tolist()))}
    return {'num_samples':n,'top1_accuracy':int(confusion.diag().sum())/n,
        'top5_accuracy':int((ranked==truth[:,None]).any(1).sum())/n,
        **{f'macro_{m}':sum(c[m] for c in per_class)/len(names) for m in ('precision','recall','f1')},
        **{f'weighted_{m}':sum(c[m]*c['support'] for c in per_class)/n for m in ('precision','recall','f1')},
        'class_names':names,'per_class':per_class,'confusion_matrix':confusion.tolist(),
        'predictions':predictions,'confidence_type':'uncalibrated_model_probability'}


def predict_image(classifier, image_path, *, top_k=5):
    """Run deterministic ensemble prediction on a single local image file."""
    if type(top_k) is not int or not 1 <= top_k <= len(classifier.class_names):
        raise ValueError('top_k must be between one and the number of classes')
    with load_rgb({'path':str(image_path)}) as image:
        batches={name:recipe.build_transform(training=False)(image).unsqueeze(0)
                 for name,recipe in classifier.preprocessing.items()}
    probabilities=classifier.predict_probabilities(batches)
    if probabilities.shape != (1,len(classifier.class_names)) or not torch.isfinite(probabilities).all():
        raise ValueError('Classifier did not return finite one-image probabilities')
    order=probabilities[0].argsort(descending=True,stable=True)[:top_k].tolist()
    return {'class_id':order[0],'class_name':classifier.class_names[order[0]],
        'probability':float(probabilities[0,order[0]]),'top_k':[{'class_id':i,'class_name':classifier.class_names[i],
        'probability':float(probabilities[0,i])} for i in order],
        'confidence_type':'uncalibrated_model_probability','device':str(classifier.device)}


def evaluate_classifier(classifier, prepared_dir, *, partition='val', batch_size=8,
                        num_workers=0, allow_test=False, output_dir=None):
    """Evaluate a loaded classifier across a full prepared split."""
    if partition not in ('train','val','test'):
        raise ValueError('partition must be train, val, or test')
    if type(allow_test) is not bool or partition=='test' and not allow_test:
        raise ValueError('Evaluating official test requires allow_test=True explicitly')
    if type(batch_size) is not int or batch_size<1 or type(num_workers) is not int or num_workers<0:
        raise ValueError('batch_size must be positive and num_workers nonnegative')
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError('Evaluation output directory already exists; preserve previous results')
    started=time.perf_counter()
    identity=validate_preflight_preparation(prepared_dir)
    if identity['manifest_sha256'] != classifier.manifest_sha256:
        raise ValueError('Classifier and preparation manifest hashes differ')
    if identity['class_names'] != classifier.class_names:
        raise ValueError('Classifier and preparation class mappings differ')
    manifest=load_classification_manifest(prepared_dir)
    source=load_cub_split('test' if partition=='test' else 'train')
    if list(source.class_names) != classifier.class_names:
        raise ValueError('Classifier and source class mappings differ')
    datasets={name:CubClassificationDataset(source,manifest,partition=partition,
        transform=classifier.preprocessing[name].build_transform(training=False)) for name in classifier.models}
    reference=next(iter(datasets.values())).rows
    expected=[(r['image_id'],r['class_id']) for r in reference]
    if any([(r['image_id'],r['class_id']) for r in d.rows] != expected for d in datasets.values()):
        raise ValueError('Component source identity alignment differs')
    loaders={name:DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,
        pin_memory=classifier.device.type=='cuda') for name,dataset in datasets.items()}
    chunks=[]; labels=[]; identifiers=[]
    for batches in zip(*loaders.values(),strict=True):
        first=batches[0]
        if any(b['image_id'] != first['image_id'] or not torch.equal(b['label'],first['label']) for b in batches[1:]):
            raise ValueError('Component batch identity alignment differs')
        chunks.append(classifier.predict_probabilities({name:b['image'] for name,b in zip(loaders,batches,strict=True)}))
        labels.append(first['label']);identifiers.extend(first['image_id'])
    probabilities,truth=torch.cat(chunks),torch.cat(labels)
    if list(zip(identifiers,truth.tolist())) != expected:
        raise ValueError('Evaluation coverage differs from the canonical partition')
    report=classification_report(probabilities,truth,classifier.class_names,identifiers)
    report.update(status='completed',complete=True,partition=partition,expected_rows=len(reference),
        manifest_sha256=classifier.manifest_sha256,device=str(classifier.device),batch_size=batch_size,
        num_workers=num_workers,component_settings=classifier.component_settings,augmentation=False,
        elapsed_seconds=time.perf_counter()-started,official_test_accessed=partition=='test')
    if output_dir is not None:
        save_evaluation(report,output_dir,probabilities=probabilities,labels=truth)
    return report


def save_evaluation(report, output_dir, *, probabilities=None, labels=None):
    """Save metrics.json, predictions.csv, evaluation.png, and optional probabilities.pt."""
    destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=False)
    if probabilities is not None:
        path=destination/'probabilities.pt'
        torch.save({'probabilities':probabilities,'labels':labels,'image_id':list(report['predictions']),
            'class_names':report['class_names'],'manifest_sha256':report['manifest_sha256'],
            'partition':report['partition']},path)
        with path.open('rb') as stream:report['probabilities_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    (destination/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    with (destination/'predictions.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(('image_id','true_label','predicted_label','probability'))
        writer.writerows((i,r['true_label'],r['predicted_label'],r['confidence']) for i,r in report['predictions'].items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    keys=('top1_accuracy','macro_precision','macro_recall','macro_f1')
    axes[0].bar(['Top-1','Precision','Recall','F1'],[100*report[k] for k in keys],color='#176b87')
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Score (%)');axes[0].set_title('Classification; precision/recall/F1 are macro averages')
    image=axes[1].imshow(report['confusion_matrix'],cmap='Blues',interpolation='nearest')
    axes[1].set(xlabel='Predicted class ID',ylabel='True class ID',title='Confusion matrix; original ordered class IDs')
    fig.colorbar(image,ax=axes[1],label='Images');fig.savefig(destination/'evaluation.png',dpi=160);plt.close(fig)
    return destination/'metrics.json'
