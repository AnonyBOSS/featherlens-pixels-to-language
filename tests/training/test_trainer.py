"""CPU arithmetic for training and evaluation; no downloaded weights or GPU claims."""
import importlib
import json
import math

import pytest
import torch
from torch import nn


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


def test_accumulation_matches_concatenated_windows_with_uneven_batches(api):
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
        nn.functional.cross_entropy(reference(values['image']), values['label']).backward()
        expected_optimizer.step()
    result = api.train_epoch(model, iter(loader), torch.optim.SGD(model.parameters(), lr=0.1),
                             'cpu', accumulation_steps=2)
    torch.testing.assert_close(model.weight, reference.weight)
    assert result['num_samples'] == 7
    assert result['optimizer_updates'] == 3
    assert result['skipped_updates'] == 0


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
