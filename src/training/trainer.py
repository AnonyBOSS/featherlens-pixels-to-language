"""Small classification loops. Accuracies and precision/recall/F1 are fractions."""
from contextlib import nullcontext

import torch
from torch import nn


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
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.grad is not None:
                parameter.grad.div_(num_samples)
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
                precision='fp32', scaler=None, progress=False) -> dict:
    """Train once over dictionary batches; the adapter owns its training policy.

    FP16 creates a GradScaler at scale 128 when none is supplied. Pass a scaler
    across epochs to preserve its state. A supplied scaler also permits CPU
    arithmetic checks. Loss and accuracy cover every sample, including skipped
    AMP updates. The loader may be an iterator without a length.
    """
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
        with _autocast(precision, device):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        count = len(labels)
        loss_sum += loss.detach().item()
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
    return {'loss': loss_sum / num_samples, 'top1_accuracy': correct / num_samples,
            'num_samples': num_samples, 'optimizer_updates': updates, 'skipped_updates': skipped}


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
            loss = nn.functional.cross_entropy(logits, labels, reduction='sum')
        probabilities = logits.float().softmax(dim=1)
        top5 = logits.topk(min(5, num_classes), dim=1).indices
        predicted = top5[:, 0]
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
