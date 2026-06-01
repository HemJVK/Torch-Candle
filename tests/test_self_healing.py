import torch_candle as torch
import numpy as np
import pytest

def test_gradient_propagation_real_anomaly():
    # Verify that when SHA is disabled, NaN propagates naturally
    torch.Tensor.enable_sha = False
    torch.set_disable_ema_estimates(False)
    torch.clear_grad_history()
    
    w = torch.Tensor([2.0], requires_grad=True)
    w.grad = torch.Tensor([float('nan')])
    
    assert np.isnan(w.grad.item())


def test_gradient_propagation_with_ema_disabled():
    # Verify that when DISABLE_EMA_ESTIMATES is True, NaN propagates naturally
    torch.Tensor.enable_sha = True
    torch.set_disable_ema_estimates(True)
    torch.clear_grad_history()
    
    w = torch.Tensor([2.0], requires_grad=True)
    w.grad = torch.Tensor([float('nan')])
    
    assert np.isnan(w.grad.item())


def test_gradient_healing_with_history():
    # Verify that when SHA is enabled and history exists, NaN is healed using EMA
    torch.Tensor.enable_sha = True
    torch.set_disable_ema_estimates(False)
    torch.clear_grad_history()
    
    w = torch.Tensor([10.0], requires_grad=True)
    
    # 1. Establish clean history
    w.grad = torch.Tensor([2.0])
    assert w.grad.item() == 2.0
    
    # 2. Inject NaN
    w.grad = torch.Tensor([float('nan')])
    
    # 3. Retrieve grad - should heal to history value (2.0)
    healed_grad = w.grad
    assert not np.isnan(healed_grad.item())
    assert healed_grad.item() == pytest.approx(2.0)


def test_propagation_through_backward():
    # Verify that numerical failures generated during backward pass propagate naturally when SHA is disabled
    torch.Tensor.enable_sha = False
    torch.set_disable_ema_estimates(False)
    torch.clear_grad_history()
    
    w = torch.Tensor([0.0], requires_grad=True)
    loss = w / w
    loss.backward()
    
    assert np.isnan(w.grad.item())
