"""Check selective fine-tuning and Mixup without external data or weights."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.resnet50 import ResNet50Classifier
import src.training.resnet50_training as training


def test_last_block_scope_updates_only_selected_encoder_parameters():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = ResNet50Classifier(weights=None, finetune_scope="last_block")
        model.set_stage("finetune")
        model.train()
        assert not model.encoder.layer4[0].training
        assert model.encoder.layer4[-1].training
        assert all(not layer.training for layer in model.encoder.modules() if isinstance(layer, nn.BatchNorm2d))
        assert sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) == 4462592
        optimizer = training.build_optimizer(
            model, classifier_learning_rate=0.0001,
            encoder_learning_rate=0.00001, weight_decay=0.01,
        )
        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert grouped == {id(p) for p in model.parameters() if p.requires_grad}
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        buffers = {name: value.clone() for name, value in model.named_buffers()}
        last_weight = model.encoder.layer4[-1].conv3.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(torch.randn(2, 3, 64, 64)), torch.tensor([0, 199]))
        loss.backward()
        optimizer.step()
        assert not torch.equal(last_weight, model.encoder.layer4[-1].conv3.weight)
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert parameter.grad is None
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        model.set_stage("head")
        assert not any(p.requires_grad for p in model.encoder.parameters())
    finally:
        torch.set_num_threads(previous_threads)


def test_mixup_uses_both_targets_and_weighted_accuracy(monkeypatch):
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    images = 2 * torch.eye(2)
    labels = torch.tensor([0, 1])
    loader = DataLoader([
        {"image": image, "label": label}
        for image, label in zip(images, labels)
    ], batch_size=2)
    monkeypatch.setattr(training.np.random, "beta", lambda *args: 0.25)
    monkeypatch.setattr(training.torch, "randperm", lambda count, **kwargs: torch.tensor([1, 0], **kwargs))
    mixed = 0.25 * images + 0.75 * images.flip(0)
    with torch.no_grad():
        logits = model(mixed)
        expected = (
            0.25 * nn.functional.cross_entropy(logits, labels)
            + 0.75 * nn.functional.cross_entropy(logits, labels.flip(0))
        ).item()
    metrics = training.train_one_epoch(
        model, loader, torch.optim.SGD(model.parameters(), lr=0.0),
        torch.device("cpu"), mixup_alpha=0.2,
    )
    assert abs(metrics.loss - expected) < 1e-6
    assert metrics.accuracy == 0.75
    assert metrics.samples == 2


@pytest.mark.parametrize("alpha", [-0.1, float("nan"), float("inf"), True])
def test_invalid_mixup_fails_before_training(alpha):
    model = nn.Linear(2, 2)
    model.eval()
    with pytest.raises(ValueError, match="mixup_alpha"):
        training.train_one_epoch(
            model, [], torch.optim.SGD(model.parameters(), lr=0.01),
            torch.device("cpu"), mixup_alpha=alpha,
        )
    assert not model.training


def test_scope_preserves_state_dictionary_compatibility():
    original = ResNet50Classifier(weights=None)
    selective = ResNet50Classifier(weights=None, finetune_scope="last_block")
    selective.load_state_dict(original.state_dict(), strict=True)
    assert original.state_dict().keys() == selective.state_dict().keys()
