"""Policy, normalization-buffer and CAM-gradient tests on real encoders."""

import pytest
import torch
from torch import nn

from .test_pretrained import api, model


def _partial_parameter(name, architecture):
    if name.startswith("head."):
        return True
    prefixes = {
        "convnext_tiny": ("encoder.6.", "encoder.7.", "encoder_norm."),
        "efficientnet_b3": ("encoder.7.", "encoder.8."),
        "regnet_y_3_2gf": ("encoder.trunk_output.block4.",),
    }
    return name.startswith(prefixes[architecture])


def test_each_policy_has_exact_stage_mask_and_survives_train_eval(model):
    for policy in ["partial", "full", "frozen", "full", "partial"]:
        model.set_fine_tuning(policy)
        for training in [True, False, True]:
            model.train(training)
            assert model.head.training is training
            for name, parameter in model.named_parameters():
                expected = policy == "full" or name.startswith("head.")
                if policy == "partial":
                    expected = _partial_parameter(name, model.config.architecture)
                assert parameter.requires_grad is expected, (policy, name)
            for module in model.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    assert not module.training


def test_newly_frozen_parameters_drop_stale_gradients(model):
    model.set_fine_tuning("full")
    model(torch.randn(2, 3, 64, 64)).square().mean().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    model.set_fine_tuning("partial")
    for parameter in model.parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
    model.set_fine_tuning("frozen")
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.encoder_norm.parameters())


def test_frozen_encoder_keeps_real_input_and_target_activation_gradients(model):
    model.set_fine_tuning("frozen")
    model.eval()
    activations = []
    def capture(module, inputs, output):
        output.retain_grad()
        activations.append(output)
    handle = model.gradcam_target_layer.register_forward_hook(capture)
    images = torch.randn(1, 3, 64, 64, requires_grad=True)
    model(images)[0, 0].backward()
    handle.remove()
    assert len(activations) == 1
    assert activations[0].ndim == 4
    assert activations[0].grad is not None
    assert torch.isfinite(activations[0].grad).all()
    assert activations[0].grad.abs().sum() > 0
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.head[1].weight.grad is not None


def test_batchnorm_buffers_respect_active_and_frozen_stages(model):
    batchnorms = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not batchnorms:
        assert model.config.architecture == "convnext_tiny"
        return
    for policy, freeze_stats in [("frozen", False), ("partial", True), ("partial", False), ("full", False)]:
        model.set_fine_tuning(policy, freeze_bn_stats=freeze_stats)
        model.train()
        before = {name: (bn.running_mean.clone(), bn.running_var.clone(), bn.num_batches_tracked.clone())
                  for name, bn in batchnorms}
        with torch.no_grad():
            model(torch.randn(2, 3, 64, 64) + 2)
        for name, bn in batchnorms:
            active = policy == "full" or (policy == "partial" and _partial_parameter(name + ".weight", model.config.architecture))
            should_update = active and not freeze_stats
            assert bn.training is should_update
            mean, var, batches = before[name]
            assert int(bn.num_batches_tracked - batches) == int(should_update)
            if should_update:
                assert not torch.equal(var, bn.running_var)
            else:
                torch.testing.assert_close(mean, bn.running_mean, rtol=0, atol=0)
                torch.testing.assert_close(var, bn.running_var, rtol=0, atol=0)
        model.eval()
        assert all(not bn.training for _, bn in batchnorms)


def test_policy_validation_does_not_partially_mutate_model(model):
    before = [p.requires_grad for p in model.parameters()]
    for arguments in [("invalid", {}), ("full", {"freeze_bn_stats": 1})]:
        with pytest.raises((ValueError, TypeError)):
            model.set_fine_tuning(arguments[0], **arguments[1])
        assert [p.requires_grad for p in model.parameters()] == before


def test_policy_omitted_bn_flag_preserves_last_explicit_choice(model):
    model.set_fine_tuning("partial", freeze_bn_stats=False)
    model.set_fine_tuning("full")
    assert model.config.freeze_bn_stats is False
    model.set_fine_tuning("frozen")
    assert model.config.fine_tuning == "frozen"
