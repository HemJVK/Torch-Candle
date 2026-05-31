import torch_candle as torch
import numpy as np
import pytest

def test_gradient_propagation_real_anomaly():
    # 1. Establish requires_grad parameter
    w = torch.Tensor([2.0], requires_grad=True)
    
    # 2. Set gradient directly to NaN (simulating autograd instability)
    w.grad = torch.Tensor([float('nan')])
    
    # Retrieving grad must propagate the NaN naturally (no healing occurs!)
    current_grad = w.grad
    assert np.isnan(current_grad.item())


def test_propagation_through_backward():
    # Verify that numerical failures generated during backward pass propagate naturally
    w = torch.Tensor([0.0], requires_grad=True)
    # 0.0 / 0.0 is NaN, so backward will propagate NaN
    loss = w / w
    loss.backward()
    
    assert np.isnan(w.grad.item())
