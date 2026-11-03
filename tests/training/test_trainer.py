"""CPU arithmetic for training and evaluation; no downloaded weights or GPU claims."""
from dataclasses import replace
import importlib
import json
import math
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from src.data.classification import make_classification_loaders
from src.models.finetuning import apply_parameter_policy, apply_training_policy
from src.models.pretrained import ModelConfig


@pytest.fixture
def api():
    assert importlib.util.find_spec('src.training.trainer') is not None, 'Training primitives are missing'
    return importlib.import_module('src.training.trainer')


def batch(images, labels, ids=None):
    return {'image': torch.tensor(images, dtype=torch.float32),
            'label': torch.tensor(labels),
            'image_id': ids if ids is not None else [f'image-{i}' for i in range(len(labels))]}


def test_train_uses_sample_weighted_loss_and_accuracy(api):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    log3 = math.log(3)
    result = api.train_epoch(model, [batch([[log3, 0], [log3, 0]], [0, 1]),
                                     batch([[0, log3]], [1])],
                             torch.optim.SGD(model.parameters(), lr=0), 'cpu')
    assert result == pytest.approx({'loss': (2 * math.log(4 / 3) + math.log(4)) / 3,
                                   'top1_accuracy': 2 / 3, 'num_samples': 3,
                                   'optimizer_updates': 2, 'skipped_updates': 0})


@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
def test_mixup_matches_explicit_soft_target_loss_gradient_and_weighted_accuracy(api, monkeypatch, label_smoothing):
    # With lambda=.25 and a swap: inputs [[4,0],[0,2]] become [[1,1.5],[3,.5]].
    samples = batch([[4, 0], [0, 2]], [0, 1], ['first', 'second'])
    original_images, original_labels = samples['image'].clone(), samples['label'].clone()
    model, reference = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2)); reference.weight.copy_(torch.eye(2))
    draws, permutations, observed_inputs = [], [], []
    def beta(a, b):
        assert a == b == 0.2
        draws.append(True)
        return 0.25
    def permutation(count, *, device=None):
        assert count == 2
        permutations.append(True)
        return torch.tensor([1, 0], device=device)
    monkeypatch.setattr(np.random, 'beta', beta)
    monkeypatch.setattr(torch, 'randperm', permutation)
    hook = model.register_forward_pre_hook(lambda module, args: observed_inputs.append(args[0].detach().clone()))
    mixed = torch.tensor([[1.0, 1.5], [3.0, 0.5]])
    targets = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    targets = targets * (1 - label_smoothing) + label_smoothing / 2
    expected_loss = -(targets * reference(mixed).log_softmax(1)).sum() / 2
    expected_loss.backward()
    expected_weight = reference.weight.detach() - 0.1 * reference.weight.grad
    result = api.train_epoch(model, [samples], torch.optim.SGD(model.parameters(), lr=0.1),
                             'cpu', mixup_alpha=0.2, label_smoothing=label_smoothing)
    hook.remove()
    assert len(draws) == len(permutations) == len(observed_inputs) == 1
    assert torch.equal(observed_inputs[0], mixed)
    assert result['loss'] == pytest.approx(expected_loss.item())
    torch.testing.assert_close(model.weight, expected_weight)
    assert result['top1_accuracy'] == 0.75
    assert result['accuracy_kind'] == 'mixup_weighted'
    assert result['num_samples'] == 2 and result['optimizer_updates'] == 1
    assert torch.equal(samples['image'], original_images) and torch.equal(samples['label'], original_labels)
    assert samples['label'].dtype == torch.int64 and samples['image_id'] == ['first', 'second']
    # Evaluation still uses the original hard labels and must never sample MixUp randomness.
    def no_mixup(*args, **kwargs):
        pytest.fail('Evaluation consumed MixUp randomness')
    monkeypatch.setattr(np.random, 'beta', no_mixup)
    monkeypatch.setattr(torch, 'randperm', no_mixup)
    with torch.no_grad():
        hard_loss = nn.functional.cross_entropy(nn.functional.linear(original_images, expected_weight), original_labels).item()
    validated = api.evaluate(model, [samples], 'cpu', class_names=['a', 'b'])
    assert validated['loss'] == pytest.approx(hard_loss)
    assert [validated['predictions'][key]['true_label'] for key in ('first', 'second')] == [0, 1]


def test_mixup_zero_preserves_legacy_numerics_keyset_and_all_rng_states(api, monkeypatch):
    samples = [batch([[1, 2], [3, 1]], [0, 1]), batch([[2, 4]], [1])]
    originals = [(value['image'].clone(), value['label'].clone(), list(value['image_id'])) for value in samples]
    def no_draw(*args, **kwargs):
        pytest.fail('Disabled MixUp consumed a random draw')
    monkeypatch.setattr(np.random, 'beta', no_draw)
    monkeypatch.setattr(torch, 'randperm', no_draw)
    reference = nn.Sequential(nn.Dropout(0.25), nn.Linear(2, 2, bias=False))
    initial = {key: value.clone() for key, value in reference.state_dict().items()}
    random.seed(321); np.random.seed(321); torch.manual_seed(321)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = correct = 0
    for value in samples:
        logits = reference(value['image'])
        loss = nn.functional.cross_entropy(logits, value['label'], reduction='sum')
        loss_sum += loss.item(); correct += int((logits.argmax(1) == value['label']).sum())
        loss.backward()
    for parameter in reference.parameters():
        parameter.grad.div_(3)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    expected = {'loss': loss_sum / 3, 'top1_accuracy': correct / 3, 'num_samples': 3,
                'optimizer_updates': 1, 'skipped_updates': 0}
    expected_python, expected_numpy, expected_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
    for options in ({}, {'mixup_alpha': 0.0}):
        model = nn.Sequential(nn.Dropout(0.25), nn.Linear(2, 2, bias=False))
        model.load_state_dict(initial)
        random.seed(321); np.random.seed(321); torch.manual_seed(321)
        result = api.train_epoch(model, samples, torch.optim.SGD(model.parameters(), lr=0.1),
                                 'cpu', accumulation_steps=2, **options)
        assert result == expected
        assert set(result) == {'loss', 'top1_accuracy', 'num_samples', 'optimizer_updates', 'skipped_updates'}
        assert all(torch.equal(value, reference.state_dict()[key]) for key, value in model.state_dict().items())
        assert random.getstate() == expected_python and torch.equal(torch.get_rng_state(), expected_torch)
        actual_numpy = np.random.get_state()
        assert actual_numpy[0] == expected_numpy[0] and np.array_equal(actual_numpy[1], expected_numpy[1])
        assert actual_numpy[2:] == expected_numpy[2:]
    for value, (images, labels, identifiers) in zip(samples, originals):
        assert torch.equal(value['image'], images) and torch.equal(value['label'], labels) and value['image_id'] == identifiers


def test_mixup_uneven_accumulation_matches_explicit_mixed_windows(api, monkeypatch):
    samples = [batch([[1], [2], [4]], [0, 1, 0]), batch([[3], [5]], [1, 0]), batch([[6]], [1])]
    lambdas, orders, counts = iter([0.2, 0.7, 0.4]), iter([[2, 0, 1], [1, 0], [0]]), []
    def beta(a, b):
        assert a == b == 0.2
        return next(lambdas)
    def permutation(count, *, device=None):
        order = next(orders); assert count == len(order)
        counts.append(count)
        return torch.tensor(order, device=device)
    monkeypatch.setattr(np.random, 'beta', beta); monkeypatch.setattr(torch, 'randperm', permutation)
    model, reference = nn.Linear(1, 2, bias=False), nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3], [-0.2]])); reference.weight.copy_(model.weight)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    windows = [(torch.tensor([[3.4], [1.2], [2.4], [3.6], [4.4]]),
                torch.tensor([[1., 0.], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3]])),
               (torch.tensor([[6.]]), torch.tensor([[0., 1.]]))]
    for images, targets in windows:
        optimizer.zero_grad()
        loss = -((targets * 0.9 + 0.05) * reference(images).log_softmax(1)).sum() / len(images)
        loss.backward(); optimizer.step()
    result = api.train_epoch(model, iter(samples), torch.optim.SGD(model.parameters(), lr=0.1), 'cpu',
                             accumulation_steps=2, mixup_alpha=0.2, label_smoothing=0.1)
    torch.testing.assert_close(model.weight, reference.weight)
    assert counts == [3, 2, 1]
    assert list(lambdas) == [] and list(orders) == []
    assert result['num_samples'] == 6 and result['optimizer_updates'] == 2 and result['skipped_updates'] == 0


def test_train_changes_parameters_by_the_mean_gradient(api):
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Stale gradients must not leak from the preceding epoch.
    model.weight.grad = torch.full_like(model.weight, 100)
    api.train_epoch(model, [batch([[1]], [0])], optimizer, 'cpu')
    torch.testing.assert_close(model.weight, torch.tensor([[0.05], [-0.05]]))
    assert model.training
    assert model.weight.grad is None


@pytest.mark.parametrize('label_smoothing', [0.0, 0.1])
def test_accumulation_matches_concatenated_windows_with_uneven_batches(api, label_smoothing):
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    reference = nn.Linear(1, 2, bias=False)
    reference.load_state_dict(model.state_dict())
    loader = [batch([[1], [2]], [0, 1]), batch([[4]], [1]),
              batch([[3], [5]], [0, 1]), batch([[2]], [0]), batch([[6]], [1])]
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for images, labels in [([[1], [2], [4]], [0, 1, 1]),
                           ([[3], [5], [2]], [0, 1, 0]), ([[6]], [1])]:
        expected_optimizer.zero_grad()
        values = batch(images, labels)
        nn.functional.cross_entropy(reference(values['image']), values['label'],
                                    label_smoothing=label_smoothing).backward()
        expected_optimizer.step()
    result = api.train_epoch(model, iter(loader), torch.optim.SGD(model.parameters(), lr=0.1),
                             'cpu', accumulation_steps=2, label_smoothing=label_smoothing)
    torch.testing.assert_close(model.weight, reference.weight)
    assert result['num_samples'] == 7
    assert result['optimizer_updates'] == 3
    assert result['skipped_updates'] == 0


def test_smoothed_training_loss_keeps_validation_cross_entropy_unsmoothed(api):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    samples = [batch([[math.log(3), 0]], [0], ['one'])]
    trained = api.train_epoch(model, samples, torch.optim.SGD(model.parameters(), lr=0),
                              'cpu', label_smoothing=0.2)
    # Uniform smoothing gives target probabilities [0.9, 0.1], not [0.8, 0.2].
    assert trained['loss'] == pytest.approx(0.9 * math.log(4 / 3) + 0.1 * math.log(4))
    validated = api.evaluate(model, samples, 'cpu', class_names=['a', 'b'])
    assert validated['loss'] == pytest.approx(math.log(4 / 3))
    assert trained['top1_accuracy'] == validated['top1_accuracy'] == 1


@pytest.mark.parametrize('overflow', [False, True])
def test_real_cpu_scaler_unscales_once_per_window_and_counts_skips(api, overflow):
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    scaler = torch.amp.GradScaler('cpu', init_scale=128)
    if overflow:
        model.weight.register_hook(lambda gradient: gradient * float('inf'))
    result = api.train_epoch(model, [batch([[1]], [0]), batch([[3]], [0])],
                             torch.optim.SGD(model.parameters(), lr=0.1), 'cpu',
                             accumulation_steps=2, scaler=scaler)
    expected = [[0], [0]] if overflow else [[0.1], [-0.1]]
    torch.testing.assert_close(model.weight, torch.tensor(expected, dtype=torch.float32))
    assert result['optimizer_updates'] == int(not overflow)
    assert result['skipped_updates'] == int(overflow)
    assert scaler.get_scale() == (64 if overflow else 128)


def test_evaluate_hand_computed_metrics_and_serializable_predictions(api):
    class InferenceOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.zeros(6))
            self.dropout = nn.Dropout(0.9)

        def forward(self, images):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
            return self.dropout(images) + self.offset

    model = InferenceOnly()
    model.offset.grad = torch.ones_like(model.offset)
    logits = [[6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1],
              [5, 6, 4, 3, 2, 1], [5, 6, 4, 3, 2, 1], [6, 5, 1, 4, 3, 2]]
    loader = [batch(logits[:3], [0, 0, 0], ['a', 'b', 'c']),
              batch(logits[3:], [1, 2], ['d', 'e'])]
    result = api.evaluate(model, loader, 'cpu', class_names=list('ABCDEF'))
    assert result['loss'] == pytest.approx(math.log(sum(math.exp(-i) for i in range(6))) + 6 / 5)
    assert result['top1_accuracy'] == pytest.approx(3 / 5)
    assert result['top5_accuracy'] == pytest.approx(4 / 5)
    assert result['macro_precision'] == pytest.approx(7 / 36)
    assert result['macro_recall'] == pytest.approx(5 / 18)
    assert result['macro_f1'] == pytest.approx(2 / 9)
    assert result['num_samples'] == 5
    assert result['confusion_matrix'] == [[2, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0], *[[0] * 6 for _ in range(3)]]
    assert result['per_class'][0] == pytest.approx({'class_id': 0, 'class_name': 'A',
                                                  'support': 3, 'precision': 2 / 3,
                                                  'recall': 2 / 3, 'f1': 2 / 3})
    assert result['per_class'][5] == {'class_id': 5, 'class_name': 'F', 'support': 0,
                                     'precision': 0, 'recall': 0, 'f1': 0}
    assert result['predictions']['c'] == pytest.approx({
        'true_label': 0, 'predicted_label': 1, 'top5': [1, 0, 2, 3, 4],
        'confidence': 1 / sum(math.exp(-i) for i in range(6))})
    assert set(result['predictions']) == set('abcde')
    json.dumps(result, allow_nan=False)
    torch.testing.assert_close(model.offset, torch.zeros(6))
    torch.testing.assert_close(model.offset.grad, torch.ones(6))


@pytest.mark.parametrize('across_batches', [False, True])
def test_evaluate_rejects_duplicate_image_ids(api, across_batches):
    loader = ([batch([[2, 1]], [0], ['same']), batch([[1, 2]], [1], ['same'])]
              if across_batches else [batch([[2, 1], [1, 2]], [0, 1], ['same', 'same'])])
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        api.evaluate(nn.Identity(), loader, 'cpu', class_names=['A', 'B'])


def test_evaluate_top5_supports_fewer_than_five_classes(api):
    result = api.evaluate(nn.Identity(), [batch([[2, 1]], [1], ['a'])],
                          'cpu', class_names=['A', 'B'])
    assert result['top5_accuracy'] == 1
    assert result['predictions']['a']['top5'] == [0, 1]


@pytest.mark.parametrize('accumulation_steps', [0, -1, True, 1.5])
def test_train_rejects_invalid_accumulation_before_updates(api, accumulation_steps):
    model = nn.Linear(1, 2)
    with pytest.raises(ValueError, match='accumulation_steps'):
        api.train_epoch(model, [batch([[1]], [0])], torch.optim.SGD(model.parameters(), lr=1),
                         'cpu', accumulation_steps=accumulation_steps)


def test_empty_loaders_fail_clearly(api):
    model = nn.Linear(1, 2)
    with pytest.raises(ValueError, match='[Ee]mpty'):
        api.train_epoch(model, [], torch.optim.SGD(model.parameters(), lr=1), 'cpu')
    with pytest.raises(ValueError, match='[Ee]mpty'):
        api.evaluate(model, [], 'cpu', class_names=['A', 'B'])


@pytest.mark.parametrize('amp', ['off', 'auto'])
def test_cpu_precision_is_fp32(api, amp):
    assert api.resolve_precision(amp, 'cpu') == 'fp32'


@pytest.mark.parametrize('amp', ['float16', 'bfloat16', 'fp8'])
def test_unsupported_cpu_precision_fails(api, amp):
    with pytest.raises(ValueError, match='precision|AMP|amp'):
        api.resolve_precision(amp, 'cpu')


@pytest.mark.parametrize('native', [False, True])
def test_cuda_auto_requires_native_bfloat16(api, monkeypatch, native):
    from contextlib import nullcontext
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device', lambda device: nullcontext())
    def native_support(*, including_emulation=True):
        assert including_emulation is False
        return native
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', native_support)
    assert api.resolve_precision('auto', 'cuda:0') == ('bfloat16' if native else 'float16')
    assert api.resolve_precision('float16', 'cuda:0') == 'float16'
    if native:
        assert api.resolve_precision('bfloat16', 'cuda:0') == 'bfloat16'
    else:
        with pytest.raises(ValueError, match='native'):
            api.resolve_precision('bfloat16', 'cuda:0')


@pytest.mark.parametrize('num_classes', [6, 1000])
def test_evaluate_ties_use_lowest_class_id_for_top1_and_ranking(api, num_classes):
    result = api.evaluate(nn.Identity(), [batch([[0] * num_classes], [0], ['tie'])],
                          'cpu', class_names=[f'C{i}' for i in range(num_classes)])
    assert result['top1_accuracy'] == 1
    assert result['predictions']['tie']['predicted_label'] == 0
    assert result['predictions']['tie']['top5'] == [0, 1, 2, 3, 4]
    assert result['predictions']['tie']['confidence'] == pytest.approx(1 / num_classes)


@pytest.mark.parametrize('precision', ['float16', 'bfloat16'])
def test_loops_reject_half_precision_on_cpu_before_forward(api, precision):
    model = nn.Linear(1, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match='precision|AMP'):
        api.train_epoch(model, [batch([[1]], [0])], optimizer, 'cpu', precision=precision)
    with pytest.raises(ValueError, match='precision|AMP'):
        api.evaluate(model, [batch([[1]], [0])], 'cpu', class_names=['A', 'B'], precision=precision)


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
@pytest.mark.parametrize('scaling', ['none', 'disabled', 'enabled'])
def test_train_rejects_nonfinite_loss_before_backward(api, bad_value, scaling):
    model = nn.Linear(1, 2, bias=False)
    nn.init.ones_(model.weight)
    before = model.weight.detach().clone()
    scaler = None if scaling == 'none' else torch.amp.GradScaler('cpu', enabled=scaling == 'enabled')
    with pytest.raises(FloatingPointError, match='[Nn]on-finite.*loss'):
        api.train_epoch(model, [batch([[bad_value]], [0])],
                        torch.optim.SGD(model.parameters(), lr=0.1), 'cpu', scaler=scaler)
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
@pytest.mark.parametrize('disabled_scaler', [False, True])
def test_train_rejects_nonfinite_unprotected_gradients(api, bad_value, disabled_scaler):
    model = nn.Linear(1, 2, bias=False)
    nn.init.zeros_(model.weight)
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, bad_value))
    scaler = torch.amp.GradScaler('cpu', enabled=False) if disabled_scaler else None
    with pytest.raises(FloatingPointError, match='[Nn]on-finite.*gradient'):
        api.train_epoch(model, [batch([[1]], [0])],
                        torch.optim.SGD(model.parameters(), lr=0.1), 'cpu', scaler=scaler)
    torch.testing.assert_close(model.weight, torch.zeros_like(model.weight))


@pytest.mark.parametrize('logits', [[float('nan'), 0], [float('inf'), 0],
                                   [0, float('-inf')], [-3e38, 3e38]])
def test_evaluate_rejects_nonfinite_outputs_or_loss(api, logits):
    with pytest.raises(FloatingPointError, match='[Nn]on-finite'):
        api.evaluate(nn.Identity(), [batch([logits], [0])], 'cpu', class_names=['A', 'B'])


def test_evaluate_rejects_nonfinite_confidence_after_float_conversion(api):
    values = batch([[0, 0]], [0])
    values['image'] = torch.tensor([[1e40, 0]], dtype=torch.float64)
    with pytest.raises(FloatingPointError, match='[Nn]on-finite'):
        api.evaluate(nn.Identity(), [values], 'cpu', class_names=['A', 'B'])

# Fit tests use synthetic 200-logit adapters, never official pretrained evidence.


CLASS_NAMES = [f'Bird {i}' for i in range(200)]
DATA_IDENTITY = {'manifest_sha256': 'a' * 64, 'class_names': CLASS_NAMES}
PROVENANCE = {'status': 'synthetic_fixture', 'architecture': 'convnext_tiny'}


class TinyAdapter(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.config = ModelConfig('convnext_tiny', head_dropout=dropout)
        self.encoder = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 4))
        self.encoder_norm = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4, 200))
        self.set_fine_tuning('frozen')

    def set_fine_tuning(self, mode, *, freeze_bn_stats=True):
        self.config = replace(self.config, fine_tuning=mode, freeze_bn_stats=freeze_bn_stats)
        apply_parameter_policy(self.encoder, self.encoder_norm, self.head, (self.encoder[2],), mode)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        apply_training_policy(self.encoder, self.encoder_norm, (self.encoder[2],),
                              self.config.fine_tuning, training=mode,
                              freeze_bn_stats=self.config.freeze_bn_stats)
        return self

    def forward(self, images):
        return self.head(self.encoder(images))


class RandomSamples(Dataset):
    def __init__(self, training):
        self.training = training

    def __len__(self):
        return 7 if self.training else 4

    def __getitem__(self, index):
        noise = random.random() + float(np.random.random()) + torch.rand(()).item() if self.training else 0
        return {'image': torch.tensor([index / 10, noise, 1.0]),
                'label': index % 3, 'image_id': f'image-{index}'}


def tiny_loaders():
    return make_classification_loaders({'train': RandomSamples(True), 'val': RandomSamples(False)},
                                       batch_size=3, seed=42)


def tiny_config(api, **kwargs):
    return api.TrainConfig(amp='off', batch_size=3, head_dropout=0.25, **kwargs)


def run_tiny(api, directory, config, **kwargs):
    return api.fit(TinyAdapter(), tiny_loaders(), config, directory,
                   data_identity=DATA_IDENTITY, weight_provenance=PROVENANCE, **kwargs)


def test_config_accepts_json_and_is_immutable(api):
    from dataclasses import FrozenInstanceError
    assert hasattr(api, 'TrainConfig'), 'TrainConfig is missing'
    config = api.TrainConfig.from_dict({'phases': [{'mode': 'partial', 'epochs': 3, 'lr': 0.02}],
                                      'normalization': [[0.1, 0.2, 0.3], [1, 1, 1]],
                                      'label_smoothing': 0.1, 'color_jitter': 0.15, 'mixup_alpha': 0.2})
    assert config.phases == (api.Phase('partial', 3, 0.02),)
    assert config == api.TrainConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert isinstance(config.normalization[0], tuple)
    assert config.label_smoothing == 0.1 and config.color_jitter == 0.15 and config.mixup_alpha == 0.2
    zero = api.TrainConfig.from_dict({'label_smoothing': 0, 'color_jitter': 0, 'mixup_alpha': 0})
    assert type(zero.label_smoothing) is type(zero.color_jitter) is type(zero.mixup_alpha) is float
    assert api.TrainConfig.from_dict({}).mixup_alpha == 0.0
    with pytest.raises(FrozenInstanceError):
        config.seed = 0


def test_training_weight_profile_is_explicit_and_rejects_other_architectures(api, tmp_path):
    from src.models.pretrained import FB22K_WEIGHT_ID
    assert api.TrainConfig().weights == ModelConfig('convnext_tiny').weights
    assert api.TrainConfig.from_dict({}).to_dict()['weights'] == ModelConfig('convnext_tiny').weights
    assert api.TrainConfig(weights=FB22K_WEIGHT_ID).to_dict()['weights'] == FB22K_WEIGHT_ID
    with pytest.raises(ValueError, match='weights'):
        api.TrainConfig(architecture='efficientnet_b3', weights=ModelConfig('convnext_tiny').weights)
    with pytest.raises(ValueError, match='model config'):
        run_tiny(api, tmp_path / 'unused', tiny_config(api, weights=FB22K_WEIGHT_ID))


def test_regularization_bounds_fail_before_training_side_effects(api):
    for field, invalid in (('label_smoothing', [-0.1, 1, True, float('nan'), float('inf')]),
                           ('color_jitter', [-0.1, 0.51, True, float('nan'), float('inf')]),
                           ('mixup_alpha', [-0.1, True, float('nan'), float('inf')])):
        for value in invalid:
            with pytest.raises(ValueError, match=field):
                api.TrainConfig(**{field: value})
    model = nn.Linear(1, 2, bias=False).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    original = model.weight.detach().clone()
    with pytest.raises(ValueError, match='label_smoothing'):
        api.train_epoch(model, [batch([[1]], [0])], optimizer, 'cpu', label_smoothing=True)
    for invalid in (-0.1, True, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='mixup_alpha'):
            api.train_epoch(model, [batch([[1]], [0])], optimizer, 'cpu', mixup_alpha=invalid)
    assert not model.training and torch.equal(model.weight, original)


@pytest.mark.parametrize('kwargs', [
    {'phases': []}, {'phases': [{'mode': 'oops'}]}, {'phases': [{'epochs': 0}]},
    {'phases': [{'lr': float('nan')}]}, {'architecture': 'resnet50'}, {'batch_size': True},
    {'accumulation_steps': 0}, {'num_workers': -1}, {'seed': 2**32}, {'seed': True},
    {'amp': 'fp8'}, {'weight_decay': -1}, {'lr_decay': 0}, {'lr_decay': float('inf')},
    {'patience': 0}, {'patience': True}, {'min_delta': -1}, {'head_dropout': 1},
    {'freeze_bn_stats': 1}, {'image_size': 0}, {'normalization': [[0] * 3, [0] * 3]},
])
def test_config_rejects_invalid_values(api, kwargs):
    assert hasattr(api, 'TrainConfig'), 'TrainConfig is missing'
    with pytest.raises((ValueError, TypeError)):
        api.TrainConfig.from_dict(kwargs)


@pytest.mark.parametrize('mixup_alpha', [0.0, 0.2])
@pytest.mark.parametrize('interrupt_after', [1, 2, 3])
def test_fit_resume_matches_uninterrupted_rng_and_phase_state(api, tmp_path, interrupt_after, mixup_alpha):
    assert hasattr(api, 'fit'), 'fit is missing'
    from src.training.checkpoint import read_checkpoint
    config = tiny_config(api, phases=(api.Phase('frozen', 2, 0.01, backbone_lr=0.001),
                                      api.Phase('partial', 2, 0.003, backbone_lr=0.0003)),
                         lr_decay=0.5, selection_metric='val_macro_f1', mixup_alpha=mixup_alpha)
    def callback(record, validation):
        random.random()
        np.random.random()
        torch.rand(3)
        if record['epoch'] == interrupt_after + 1:
            raise RuntimeError('interrupted')
    torch.manual_seed(19)
    run_tiny(api, tmp_path / 'full', config,
             on_epoch=lambda record, validation: (random.random(), np.random.random(), torch.rand(3)))
    torch.manual_seed(19)
    with pytest.raises(RuntimeError, match='interrupted'):
        run_tiny(api, tmp_path / 'split', config, on_epoch=callback)
    # New random construction and external RNG activity must not affect continuation.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    result = run_tiny(api, tmp_path / 'split', config, resume=True,
                      on_epoch=lambda record, validation: (random.random(), np.random.random(), torch.rand(3)))
    full = read_checkpoint(tmp_path / 'full' / 'latest.pt')
    split = read_checkpoint(tmp_path / 'split' / 'latest.pt')
    assert result['history'] == full['history']
    assert [row['learning_rate'] for row in result['history']] == [0.01, 0.005, 0.003, 0.0015]
    assert [row['backbone_learning_rate'] for row in result['history']] == [None, None, 0.0003, 0.00015]
    assert len(split['optimizer_state']['param_groups']) == 2
    assert result['optimizer_updates'] == 8
    for name, tensor in full['model_state'].items():
        assert torch.equal(tensor, split['model_state'][name]), name
    assert split['scheduler_state'] == full['scheduler_state']
    for state in split['optimizer_state']['state'].values():
        assert state['step'].item() == 4  # two epochs in new phase, no carried Adam moments
    assert torch.equal(split['rng_state']['torch_cpu'], full['rng_state']['torch_cpu'])
    assert split['rng_state']['python'] == full['rng_state']['python']
    a, b = split['rng_state']['numpy'], full['rng_state']['numpy']
    assert a[0] == b[0] and torch.equal(a[1], b[1]) and a[2:] == b[2:]
    assert all(torch.equal(state, full['rng_state']['loader_generators'][name])
               for name, state in split['rng_state']['loader_generators'].items())
    if mixup_alpha:
        assert all(row['train_accuracy_kind'] == 'mixup_weighted' for row in result['history'])
    else:
        assert all('train_accuracy_kind' not in row for row in result['history'])
    assert result['validation'] == full['validation']


def test_fit_phase_masks_and_fresh_adam_state(api, tmp_path):
    assert hasattr(api, 'fit'), 'fit is missing'
    model = TinyAdapter()
    config = tiny_config(api, phases=tuple(api.Phase(mode, 1, 0.01) for mode in ('frozen', 'partial', 'full')))
    snapshots = []
    def capture(record, validation):
        snapshots.append({name: parameter.detach().clone() for name, parameter in model.named_parameters()})
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    api.fit(model, tiny_loaders(), config, tmp_path, data_identity=DATA_IDENTITY,
            weight_provenance=PROVENANCE, on_epoch=capture)
    for name in before:
        changed = [not torch.equal(a[name], b[name]) for a, b in zip([before] + snapshots, snapshots)]
        expected = [True, True, True] if name.startswith('head') else (
            [False, True, True] if name.startswith('encoder.2') else [False, False, True])
        assert changed == expected, name
    from src.training.checkpoint import read_checkpoint
    payload = read_checkpoint(tmp_path / 'latest.pt')
    assert all(value['step'].item() == 2 for value in payload['optimizer_state']['state'].values())
    assert payload['model_config']['fine_tuning'] == 'full'


def test_best_strict_ties_and_independent_early_stop_monitor(api, tmp_path, monkeypatch):
    assert hasattr(api, 'fit'), 'fit is missing'
    evaluate = api.evaluate
    losses = iter([5.0, 5.0, 4.95, 4.94, 3.0])
    def prescribed_loss(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        result['loss'] = next(losses)
        return result
    monkeypatch.setattr(api, 'evaluate', prescribed_loss)
    config = tiny_config(api, phases=(api.Phase('frozen', 5),), patience=3, min_delta=0.1)
    result = run_tiny(api, tmp_path, config)
    assert result['epochs_completed'] == 4
    assert result['stop_reason'] == 'early_stopping'
    assert [row['is_best'] for row in result['history']] == [True, False, True, True]
    assert result['best_epoch'] == 4
    from src.training.checkpoint import read_checkpoint
    best = read_checkpoint(tmp_path / 'best.pt')
    assert best['best_loss'] == 4.94
    assert best['early_stop_best'] == 5.0
    assert best['bad_epochs'] == 3
    assert best['stop_reason'] == 'early_stopping'
    # The exhausted loss iterator catches any unintended resumed validation.
    assert run_tiny(api, tmp_path, config, resume=True) == result


def test_completed_run_is_idempotent_and_fresh_run_cannot_overwrite(api, tmp_path):
    assert hasattr(api, 'fit'), 'fit is missing'
    config = tiny_config(api, phases=(api.Phase(epochs=1),))
    result = run_tiny(api, tmp_path, config)
    assert result['stop_reason'] == 'completed'
    before = (tmp_path / 'latest.pt').read_bytes()
    assert run_tiny(api, tmp_path, config, resume=True) == result
    assert before == (tmp_path / 'latest.pt').read_bytes()
    with pytest.raises(FileExistsError):
        run_tiny(api, tmp_path, config)


def test_fit_rejects_resume_configuration_and_data_mismatch(api, tmp_path):
    assert hasattr(api, 'fit'), 'fit is missing'
    config = tiny_config(api, phases=(api.Phase(epochs=1),))
    run_tiny(api, tmp_path, config)
    with pytest.raises(ValueError, match='config'):
        run_tiny(api, tmp_path, replace(config, lr_decay=0.9), resume=True)
    with pytest.raises(ValueError, match='data'):
        api.fit(TinyAdapter(), tiny_loaders(), config, tmp_path, resume=True,
                data_identity={**DATA_IDENTITY, 'manifest_sha256': 'b' * 64}, weight_provenance=PROVENANCE)


def test_fit_rejects_zero_successful_updates_without_checkpoint(api, tmp_path, monkeypatch):
    assert hasattr(api, 'fit'), 'fit is missing'
    train = api.train_epoch
    def no_updates(*args, **kwargs):
        return {**train(*args, **kwargs), 'optimizer_updates': 0}
    monkeypatch.setattr(api, 'train_epoch', no_updates)
    with pytest.raises(RuntimeError, match='optimizer updates'):
        run_tiny(api, tmp_path, tiny_config(api))
    assert not (tmp_path / 'latest.pt').exists()


def test_explicit_same_mode_phase_starts_fresh_optimizer_and_schedule(api, tmp_path):
    from src.training.checkpoint import read_checkpoint
    config = tiny_config(api, phases=(api.Phase('frozen', 2, 0.01), api.Phase('frozen', 1, 0.01)),
                         lr_decay=0.5)
    result = run_tiny(api, tmp_path, config)
    assert [row['learning_rate'] for row in result['history']] == [0.01, 0.005, 0.01]
    assert all(state['step'].item() == 2 for state in read_checkpoint(tmp_path / 'latest.pt')['optimizer_state']['state'].values())


def test_fit_rejects_changed_resolved_precision_before_any_updates(api, tmp_path, monkeypatch):
    config = tiny_config(api, phases=(api.Phase(epochs=1),))
    run_tiny(api, tmp_path, config)
    monkeypatch.setattr(api, 'resolve_precision', lambda amp, device: 'float16')
    with pytest.raises(ValueError, match='precision'):
        run_tiny(api, tmp_path, config, resume=True)


@pytest.mark.parametrize('mismatch', ['workers', 'batch_size', 'persistent_workers', 'class_mapping'])
def test_fit_rejects_incompatible_loader_before_updates(api, tmp_path, mismatch):
    from types import SimpleNamespace
    loaders = tiny_loaders()
    if mismatch == 'class_mapping':
        loaders['train'].dataset.source = SimpleNamespace(class_names=list(reversed(CLASS_NAMES)))
    else:
        attribute = 'num_workers' if mismatch == 'workers' else mismatch
        object.__setattr__(loaders['train'], attribute, {'workers': 1, 'batch_size': 4, 'persistent_workers': True}[mismatch])
    with pytest.raises(ValueError, match='Loader|class mapping'):
        api.fit(TinyAdapter(), loaders, tiny_config(api), tmp_path,
                data_identity=DATA_IDENTITY, weight_provenance=PROVENANCE)
    assert not (tmp_path / 'latest.pt').exists()


def test_fit_rejects_missing_manifest_identity_before_updating_weights(api, tmp_path):
    model = TinyAdapter()
    before = model.head[1].weight.detach().clone()
    with pytest.raises(ValueError, match='manifest'):
        api.fit(model, tiny_loaders(), tiny_config(api), tmp_path,
                data_identity={'class_names': CLASS_NAMES}, weight_provenance=PROVENANCE)
    assert torch.equal(before, model.head[1].weight)


@pytest.mark.parametrize('value', [0, -0.1, True, float('nan'), float('inf')])
def test_phase_rejects_invalid_backbone_lr(api, value):
    with pytest.raises(ValueError, match='backbone_lr'):
        api.Phase(backbone_lr=value)


def test_config_rejects_unknown_selection_metric(api):
    with pytest.raises(ValueError, match='selection_metric'):
        tiny_config(api, selection_metric='val_accuracy')


def test_fit_separate_rates_update_head_and_only_unfrozen_backbone(api, tmp_path, monkeypatch):
    model = TinyAdapter(dropout=0)
    model.encoder_norm = nn.LayerNorm(4)
    model.forward = lambda images: model.head(model.encoder_norm(model.encoder(images)))
    config = api.TrainConfig(amp='off', batch_size=7, accumulation_steps=1, weight_decay=0,
                            phases=tuple(api.Phase(mode, 1, 0.01, backbone_lr=0.001)
                                         for mode in ('frozen', 'partial', 'full')))
    loaders = make_classification_loaders({'train': RandomSamples(True), 'val': RandomSamples(False)},
                                         batch_size=7, seed=42)
    train = api.train_epoch
    seen = []
    def inspect_updates(model, loader, optimizer, device, **kwargs):
        groups = [{id(p) for p in group['params']} for group in optimizer.param_groups]
        assert groups[0] == {id(p) for p in model.head.parameters()}
        backbone = {id(p) for name, p in model.named_parameters()
                    if not name.startswith('head.') and p.requires_grad}
        assert groups[1:] == ([backbone] if backbone else [])
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        result = train(model, loader, optimizer, device, **kwargs)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                expected = 0.01 if name.startswith('head.') else 0.001
                assert (parameter - before[name]).abs().max().item() == pytest.approx(expected, abs=1e-5)
            else:
                assert torch.equal(parameter, before[name]), name
        seen.append(model.config.fine_tuning)
        return result
    monkeypatch.setattr(api, 'train_epoch', inspect_updates)
    result = api.fit(model, loaders, config, tmp_path, data_identity=DATA_IDENTITY,
                     weight_provenance=PROVENANCE)
    assert seen == ['frozen', 'partial', 'full']
    assert [row['backbone_learning_rate'] for row in result['history']] == [None, 0.001, 0.001]


def test_fit_f1_selects_earliest_best_and_uses_f1_delta_for_stopping(api, tmp_path, monkeypatch):
    evaluate = api.evaluate
    metrics = iter([(5.0, 0.4), (4.0, 0.5), (3.0, 0.5), (2.0, 0.52), (1.0, 0.51)])
    def prescribed_metrics(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        result['loss'], result['macro_f1'] = next(metrics)
        return result
    monkeypatch.setattr(api, 'evaluate', prescribed_metrics)
    config = tiny_config(api, phases=(api.Phase(epochs=8),), selection_metric='val_macro_f1',
                         patience=3, min_delta=0.05)
    result = run_tiny(api, tmp_path, config)
    assert result['epochs_completed'] == 5
    assert result['stop_reason'] == 'early_stopping'
    assert result['best_epoch'] == 4
    assert result['best_loss'] == 1.0
    assert [row['is_best'] for row in result['history']] == [True, True, False, True, False]
    from src.training.checkpoint import read_checkpoint
    best = read_checkpoint(tmp_path / 'best.pt')
    latest = read_checkpoint(tmp_path / 'latest.pt')
    assert best['epochs_completed'] == 4
    assert best['best_epoch'] == 4
    assert latest['early_stop_best'] == 0.5
    assert latest['bad_epochs'] == 3
    assert run_tiny(api, tmp_path, config, resume=True) == result
