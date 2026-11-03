"""Small classification loops. Accuracies and precision/recall/F1 are fractions."""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from src.data.transforms import IMAGENET_NORMALIZATION
from src.models.pretrained import ModelConfig
from src.training.data import PreprocessingConfig


def resolve_precision(amp, device) -> str:
    """Choose CUDA AMP using native BF16 support; CPU auto stays FP32."""
    device = torch.device(device)
    if amp not in ('off', 'auto', 'float16', 'bfloat16'):
        raise ValueError('amp must be off, auto, float16 or bfloat16')
    if amp == 'off':
        return 'fp32'
    if device.type == 'cpu':
        if amp == 'auto':
            return 'fp32'
        raise ValueError('Half precision AMP requires a CUDA device')
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('AMP requires an available CUDA device')
    if amp == 'float16':
        return 'float16'
    with torch.cuda.device(device):
        native_bfloat16 = torch.cuda.is_bf16_supported(including_emulation=False)
    if amp == 'bfloat16' and not native_bfloat16:
        raise ValueError('CUDA device lacks native bfloat16 support')
    return 'bfloat16' if native_bfloat16 else 'float16'


def _autocast(precision, device):
    # Public loops accept resolved precision, not the configuration's auto mode.
    if precision not in ('fp32', 'float16', 'bfloat16'):
        raise ValueError('precision must be fp32, float16 or bfloat16')
    resolve_precision('off' if precision == 'fp32' else precision, device)
    if precision == 'fp32':
        return nullcontext()
    dtype = torch.float16 if precision == 'float16' else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _step(optimizer, scaler, num_samples):
    """Normalize summed, unscaled gradients by this window's actual samples."""
    if scaler is not None:
        scaler.unscale_(optimizer)
    protected = scaler is not None and scaler.is_enabled()
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None:
                parameter.grad.div_(num_samples)
                if not protected and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Non-finite gradients before optimizer update')
    if scaler is None:
        optimizer.step()
        successful = True
    else:
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        successful = scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    return successful


def train_epoch(model, loader, optimizer, device, *, accumulation_steps=1,
                precision='fp32', scaler=None, progress=False, label_smoothing=0.0, mixup_alpha=0.0) -> dict:
    """Train once over dictionary batches; the adapter owns its training policy.

    FP16 creates a GradScaler at scale 128 when none is supplied. Pass a scaler
    across epochs to preserve its state. A supplied scaler also permits CPU
    arithmetic checks. Loss and accuracy cover every sample, including skipped
    AMP updates. The loader may be an iterator without a length.
    Training loss uses the optional smoothed objective; evaluation stays unsmoothed.
    Positive MixUp alpha mixes within each microbatch; its accuracy is MixUp-weighted.
    """
    if (type(label_smoothing) not in (int, float) or not math.isfinite(label_smoothing)
            or not 0 <= label_smoothing < 1):
        raise ValueError('label_smoothing must be finite and between 0 (inclusive) and 1 (exclusive)')
    if type(mixup_alpha) not in (int, float) or not math.isfinite(mixup_alpha) or mixup_alpha < 0:
        raise ValueError('mixup_alpha must be finite and nonnegative')
    if type(accumulation_steps) is not int or accumulation_steps < 1:
        raise ValueError('accumulation_steps must be a positive integer')
    device = torch.device(device)
    _autocast(precision, device)
    if precision == 'float16' and scaler is None:
        scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = correct = num_samples = window_samples = window_batches = 0
    updates = skipped = 0
    for batch_index, batch in enumerate(loader, 1):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        if mixup_alpha > 0:
            coefficient = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(len(labels), device=device)
            other_labels = labels[permutation]
            images = coefficient * images + (1 - coefficient) * images[permutation]
        with _autocast(precision, device):
            logits = model(images)
            loss = nn.functional.cross_entropy(
                logits, labels, reduction='sum', label_smoothing=label_smoothing)
            if mixup_alpha > 0:
                loss = coefficient * loss + (1 - coefficient) * nn.functional.cross_entropy(
                    logits, other_labels, reduction='sum', label_smoothing=label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        count = len(labels)
        loss_sum += loss.detach().item()
        if mixup_alpha > 0:
            predicted = logits.detach().argmax(dim=1)
            correct += (coefficient * (predicted == labels).sum().item()
                        + (1 - coefficient) * (predicted == other_labels).sum().item())
        else:
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        num_samples += count
        window_samples += count
        window_batches += 1
        if window_batches == accumulation_steps:
            successful = _step(optimizer, scaler, window_samples)
            updates += int(successful)
            skipped += int(not successful)
            window_samples = window_batches = 0
        if progress:
            print(f'Batch {batch_index}: samples={num_samples}, loss={loss_sum / num_samples:.4f}', flush=True)
    if not num_samples:
        raise ValueError('Empty training loader')
    if window_batches:
        successful = _step(optimizer, scaler, window_samples)
        updates += int(successful)
        skipped += int(not successful)
    result = {'loss': loss_sum / num_samples, 'top1_accuracy': correct / num_samples,
              'num_samples': num_samples, 'optimizer_updates': updates, 'skipped_updates': skipped}
    if mixup_alpha > 0:
        result['accuracy_kind'] = 'mixup_weighted'
    return result


@torch.inference_mode()
def evaluate(model, loader, device, *, class_names, precision='fp32') -> dict:
    """Evaluate all declared classes, including absent classes in macro scores.

    Confusion rows are true labels, columns are predictions; label values and
    top5 entries are zero-based class IDs. Predictions require unique image IDs.
    """
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError('class_names must be a nonempty unique ordered sequence')
    device = torch.device(device)
    _autocast(precision, device)
    model.eval()
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    loss_sum = num_samples = top5_correct = 0
    predictions = {}
    for batch in loader:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        identifiers = batch['image_id']
        if len(identifiers) != len(labels):
            raise ValueError('Each sample must have an image_id')
        with _autocast(precision, device):
            logits = model(images)
            if logits.ndim != 2 or logits.shape != (len(labels), num_classes):
                raise ValueError('Logits must match batch size and class_names')
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Non-finite evaluation logits')
            loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite evaluation loss')
        probabilities = logits.float().softmax(dim=1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError('Non-finite evaluation probabilities')
        # Stable ordering breaks ties by lower class ID, consistent with argmax.
        top5 = logits.argsort(dim=1, descending=True, stable=True)[:, :min(5, num_classes)]
        predicted = logits.argmax(dim=1)
        top5_correct += (top5 == labels[:, None]).any(dim=1).sum().item()
        loss_sum += loss.item()
        num_samples += len(labels)
        pairs = (labels * num_classes + predicted).cpu()
        confusion += torch.bincount(pairs, minlength=num_classes**2).reshape(num_classes, num_classes)
        for image_id, label, prediction, ranked, confidence in zip(
            identifiers, labels.tolist(), predicted.tolist(), top5.tolist(),
            probabilities.gather(1, predicted[:, None]).flatten().tolist(),
        ):
            if not isinstance(image_id, str) or not image_id:
                raise ValueError('image_id must be a nonempty string')
            if image_id in predictions:
                raise ValueError(f'Duplicate image_id: {image_id}')
            predictions[image_id] = {'true_label': label, 'predicted_label': prediction,
                                     'top5': ranked, 'confidence': confidence}
    if not num_samples:
        raise ValueError('Empty evaluation loader')
    per_class = []
    for class_id, class_name in enumerate(class_names):
        true_positive = confusion[class_id, class_id].item()
        support = confusion[class_id].sum().item()
        predicted_count = confusion[:, class_id].sum().item()
        precision_value = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * true_positive / (support + predicted_count) if support + predicted_count else 0.0
        per_class.append({'class_id': class_id, 'class_name': class_name, 'support': support,
                          'precision': precision_value, 'recall': recall, 'f1': f1})
    return {'loss': loss_sum / num_samples,
            'top1_accuracy': confusion.diag().sum().item() / num_samples,
            'top5_accuracy': top5_correct / num_samples, 'num_samples': num_samples,
            **{f'macro_{metric}': sum(row[metric] for row in per_class) / num_classes
               for metric in ('precision', 'recall', 'f1')},
            'per_class': per_class, 'confusion_matrix': confusion.tolist(), 'predictions': predictions}


@dataclass(frozen=True)
class Phase:
    mode: str = 'frozen'
    epochs: int = 2
    lr: float = 0.001
    backbone_lr: float | None = None

    def __post_init__(self):
        if self.mode not in ('frozen', 'partial', 'full'):
            raise ValueError('phase mode must be frozen, partial or full')
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError('phase epochs must be a positive integer')
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError('phase lr must be finite and positive')
        if self.backbone_lr is not None and (type(self.backbone_lr) not in (int, float)
                or not math.isfinite(self.backbone_lr) or self.backbone_lr <= 0):
            raise ValueError('phase backbone_lr must be finite and positive or None')


@dataclass(frozen=True)
class TrainConfig:
    architecture: str = 'convnext_tiny'
    phases: tuple[Phase, ...] = (Phase(),)
    batch_size: int = 8
    accumulation_steps: int = 2
    weight_decay: float = 0.0001
    amp: str = 'float16'
    seed: int = 42
    num_workers: int = 0
    freeze_bn_stats: bool = True
    head_dropout: float = 0.0
    lr_decay: float = 1.0
    patience: int | None = None
    min_delta: float = 0.0
    image_size: int = 224
    normalization: tuple | None = IMAGENET_NORMALIZATION
    selection_metric: str = 'val_loss'
    label_smoothing: float = 0.0
    color_jitter: float = 0.0
    weights: str | None = None
    mixup_alpha: float = 0.0

    def __post_init__(self):
        model_config = ModelConfig(self.architecture, weights=self.weights,
                                   head_dropout=self.head_dropout, freeze_bn_stats=self.freeze_bn_stats)
        object.__setattr__(self, 'weights', model_config.weights)
        if not isinstance(self.phases, (tuple, list)) or not self.phases or any(
                not isinstance(phase, Phase) for phase in self.phases):
            raise ValueError('phases must be a nonempty sequence of Phase values')
        object.__setattr__(self, 'phases', tuple(self.phases))
        for name, minimum in (('batch_size', 1), ('accumulation_steps', 1), ('num_workers', 0), ('seed', 0)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.seed >= 2**32:
            raise ValueError('seed must be below 2**32')
        if self.amp not in ('off', 'auto', 'float16', 'bfloat16'):
            raise ValueError('amp must be off, auto, float16 or bfloat16')
        for name in ('weight_decay', 'min_delta', 'lr_decay', 'mixup_alpha'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        object.__setattr__(self, 'mixup_alpha', float(self.mixup_alpha) if self.mixup_alpha else 0.0)
        if self.lr_decay == 0:
            raise ValueError('lr_decay must be positive')
        if self.patience is not None and (type(self.patience) is not int or self.patience < 1):
            raise ValueError('patience must be a positive integer or None')
        if self.selection_metric not in ('val_loss', 'val_macro_f1'):
            raise ValueError('selection_metric must be val_loss or val_macro_f1')
        if (type(self.label_smoothing) not in (int, float) or not math.isfinite(self.label_smoothing)
                or not 0 <= self.label_smoothing < 1):
            raise ValueError('label_smoothing must be finite and between 0 (inclusive) and 1 (exclusive)')
        object.__setattr__(self, 'label_smoothing', float(self.label_smoothing) if self.label_smoothing else 0.0)
        preprocessing = PreprocessingConfig(self.image_size, self.normalization, self.color_jitter)
        object.__setattr__(self, 'normalization', preprocessing.normalization)
        object.__setattr__(self, 'color_jitter', preprocessing.color_jitter)

    def to_dict(self):
        """JSON-normalized settings, including lists for immutable tuple fields."""
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if 'phases' in values:
            values['phases'] = tuple(Phase(**phase) for phase in values['phases'])
        return cls(**values)


def fit(model, loaders, config, run_dir, *, data_identity, weight_provenance,
        device='cpu', resume=False, on_epoch=None) -> dict:
    """Fit completed epochs; latest.pt is the recovery authority.

    Every explicit phase starts fresh AdamW moments and a new StepLR, even when
    consecutive phases have the same mode. The FP16 scaler spans phases. Seeds
    cover training randomness; callers seed before constructing a fresh model.
    Nonpersistent canonical loaders are required for exact epoch continuation.
    """
    from src.training.checkpoint import (
        capture_rng_state, restore_rng_state, save_checkpoint, read_checkpoint, _load_model_state,
    )
    if not isinstance(config, TrainConfig):
        raise TypeError('config must be a TrainConfig')
    device = torch.device(device)
    precision = resolve_precision(config.amp, device)
    if set(loaders) != {'train', 'val'}:
        raise ValueError('fit requires only train and val loaders')
    manifest = data_identity.get('manifest_sha256')
    if not isinstance(manifest, str) or len(manifest) != 64 or any(c not in '0123456789abcdef' for c in manifest):
        raise ValueError('data_identity requires a manifest_sha256 digest')
    class_names = data_identity.get('class_names')
    if (not isinstance(class_names, (list, tuple)) or len(class_names) != 200
            or any(not isinstance(name, str) or not name for name in class_names)
            or len(set(class_names)) != 200):
        raise ValueError('data_identity requires the ordered 200-class mapping in class_names')
    for name, loader in loaders.items():
        if (loader.num_workers != config.num_workers or loader.persistent_workers
                or loader.batch_size != config.batch_size or loader.drop_last
                or not isinstance(loader.generator, torch.Generator)):
            raise ValueError('Loader workers, batch size and generators must match config; use nonpersistent loaders')
        source = getattr(loader.dataset, 'source', None)
        if source is not None and list(source.class_names) != list(class_names):
            raise ValueError(f'{name} loader class mapping differs from data_identity')
    if (model.config.architecture != config.architecture or model.config.weights != config.weights
            or model.config.head_dropout != config.head_dropout
            or model.config.freeze_bn_stats != config.freeze_bn_stats):
        raise ValueError('model config does not match training config')
    run_dir = Path(run_dir)
    latest_path, best_path = run_dir / 'latest.pt', run_dir / 'best.pt'
    saved = None
    if resume:
        saved = read_checkpoint(latest_path, expected_config=config.to_dict(), expected_data_identity=data_identity)
        if saved['resolved_precision'] != precision:
            raise ValueError('Resolved precision differs from checkpoint')
        if saved['weight_provenance'] != weight_provenance:
            raise ValueError('Weight provenance differs from checkpoint')
        _load_model_state(model, saved['model_state'])
        model.set_fine_tuning(saved['model_config']['fine_tuning'], freeze_bn_stats=config.freeze_bn_stats)
    elif latest_path.exists() or best_path.exists():
        raise FileExistsError('Run checkpoints already exist; use resume=True or a fresh run directory')
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    history = [] if saved is None else saved['history']
    epochs_completed = 0 if saved is None else saved['epochs_completed']
    total_updates = 0 if saved is None else saved['optimizer_updates']
    best_loss = float('inf') if saved is None else saved['best_loss']
    best_epoch = None if saved is None else saved['best_epoch']
    maximize = config.selection_metric == 'val_macro_f1'
    initial_best = -float('inf') if maximize else float('inf')
    best_metric = initial_best if best_epoch is None else history[best_epoch - 1][config.selection_metric]
    early_stop_best = initial_best if saved is None else saved['early_stop_best']
    bad_epochs = 0 if saved is None else saved['bad_epochs']
    stop_reason = None if saved is None else saved['stop_reason']
    validation = None if saved is None else saved['validation']

    def result():
        return {'epochs_completed': epochs_completed, 'best_epoch': best_epoch, 'best_loss': best_loss,
                'optimizer_updates': total_updates, 'history': history, 'stop_reason': stop_reason,
                'resolved_precision': precision, 'latest_checkpoint': str(latest_path),
                'best_checkpoint': str(best_path), 'validation': validation}

    if stop_reason is not None:
        model.eval()
        return result()
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0) if precision == 'float16' else None
    if saved is not None and scaler is not None:
        scaler.load_state_dict(saved['scaler_state'])
    start_phase = 0 if saved is None else saved['phase_index']
    start_epoch = 0 if saved is None else saved['epoch_in_phase']
    if start_epoch == config.phases[start_phase].epochs:
        start_phase += 1
        start_epoch = 0
    # Restore once, after random construction and optimizer/scaler restoration.
    restore_pending = True
    for phase_index in range(start_phase, len(config.phases)):
        phase = config.phases[phase_index]
        model.set_fine_tuning(phase.mode, freeze_bn_stats=config.freeze_bn_stats)
        head_parameters = [p for p in model.head.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head_parameters}
        backbone_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        if phase.backbone_lr is None:
            # Preserve the legacy parameter order for existing optimizer checkpoints.
            parameters = [p for p in model.parameters() if p.requires_grad]
        else:
            parameters = [{'params': head_parameters, 'lr': phase.lr}]
            if backbone_parameters:
                parameters.append({'params': backbone_parameters, 'lr': phase.backbone_lr})
        optimizer = torch.optim.AdamW(parameters, lr=phase.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.lr_decay)
        if saved is not None and phase_index == saved['phase_index']:
            optimizer.load_state_dict(saved['optimizer_state'])
            scheduler.load_state_dict(saved['scheduler_state'])
        if restore_pending:
            if saved is None:
                random.seed(config.seed)
                np.random.seed(config.seed)
                torch.manual_seed(config.seed)
            else:
                restore_rng_state(saved['rng_state'], loaders)
            restore_pending = False
        for epoch_in_phase in range(start_epoch + 1, phase.epochs + 1):
            learning_rate = optimizer.param_groups[0]['lr']
            backbone_learning_rate = optimizer.param_groups[-1]['lr'] if backbone_parameters else None
            training = train_epoch(model, loaders['train'], optimizer, device,
                                   accumulation_steps=config.accumulation_steps, precision=precision, scaler=scaler,
                                   label_smoothing=config.label_smoothing, mixup_alpha=config.mixup_alpha)
            if training['optimizer_updates'] == 0:
                raise RuntimeError('Epoch completed with zero successful optimizer updates')
            validation = evaluate(model, loaders['val'], device, class_names=class_names, precision=precision)
            epochs_completed += 1
            total_updates += training['optimizer_updates']
            metric = validation[config.selection_metric.removeprefix('val_')]
            is_best = metric > best_metric if maximize else metric < best_metric
            best_loss = min(best_loss, validation['loss'])
            if is_best:
                best_metric, best_epoch = metric, epochs_completed
            improvement = (metric > early_stop_best + config.min_delta if maximize
                           else metric < early_stop_best - config.min_delta)
            if improvement:
                early_stop_best, bad_epochs = metric, 0
            else:
                bad_epochs += 1
            if config.patience is not None and bad_epochs >= config.patience:
                stop_reason = 'early_stopping'
            elif epochs_completed == sum(p.epochs for p in config.phases):
                stop_reason = 'completed'
            record = {'epoch': epochs_completed, 'phase_index': phase_index, 'phase_mode': phase.mode,
                      'epoch_in_phase': epoch_in_phase, 'learning_rate': learning_rate,
                      'backbone_learning_rate': backbone_learning_rate,
                      'train_loss': training['loss'], 'train_top1_accuracy': training['top1_accuracy'],
                      **{f'val_{key}': validation[key] for key in (
                          'loss', 'top1_accuracy', 'top5_accuracy', 'macro_precision', 'macro_recall', 'macro_f1')},
                      'optimizer_updates': training['optimizer_updates'], 'skipped_updates': training['skipped_updates'],
                      'total_optimizer_updates': total_updates, 'num_train_samples': training['num_samples'],
                      'num_val_samples': validation['num_samples'], 'is_best': is_best}
            if config.mixup_alpha > 0:
                record['train_accuracy_kind'] = training['accuracy_kind']
            history.append(record)
            scheduler.step()
            if on_epoch is not None:
                on_epoch(record, validation)
            payload = {'schema_version': 1, 'model_state': model.state_dict(), 'model_config': asdict(model.config),
                       'training_config': config.to_dict(), 'weight_provenance': weight_provenance,
                       'data_identity': data_identity, 'class_names': list(class_names),
                       'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
                       'scaler_state': None if scaler is None else scaler.state_dict(),
                       'epochs_completed': epochs_completed, 'phase_index': phase_index, 'epoch_in_phase': epoch_in_phase,
                       'resolved_precision': precision, 'optimizer_updates': total_updates, 'history': history,
                       'best_loss': best_loss, 'best_epoch': best_epoch, 'early_stop_best': early_stop_best,
                       'bad_epochs': bad_epochs, 'stop_reason': stop_reason, 'validation': validation,
                       'rng_state': capture_rng_state(loaders)}
            if is_best:
                save_checkpoint(best_path, payload)
            save_checkpoint(latest_path, payload)
            if stop_reason is not None:
                return result()
        start_epoch = 0
    return result()
