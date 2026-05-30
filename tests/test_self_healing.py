import torch_candle as torch
import numpy as np

def test_self_healing_real_anomaly():
    # 1. Establish normal gradient to build history
    w = torch.Tensor([2.0], requires_grad=True)
    loss1 = w * 2.0
    loss1.backward()
    
    # Access grad to populate history (valid grad = 2.0)
    grad1 = w.grad
    assert grad1.item() == 2.0
    
    # 2. Simulate mathematical anomaly on a new parameter
    w2 = torch.Tensor([2.0], requires_grad=True)
    
    # Map w2's id to w1's valid gradient history to simulate a continuous step
    if not hasattr(torch.Tensor, "_grad_history"):
        torch.Tensor._grad_history = {}
    torch.Tensor._grad_history[id(w2)] = (np.array([2.0]).shape, np.array([2.0]))
    
    # Set gradient directly to NaN (simulating autograd instability)
    w2.grad = torch.Tensor([float('nan')])
    
    # Retrieving grad triggers the Self-Healing Autograd engine!
    healed_grad = w2.grad
    
    # Verify that the NaN element is completely healed and restored to 2.0!
    assert not np.isnan(healed_grad.item())
    assert healed_grad.item() == 2.0

def test_standard_mode_context_manager():
    # 1. Establish history
    w = torch.Tensor([2.0], requires_grad=True)
    loss1 = w * 2.0
    loss1.backward()
    
    # Access grad to populate history (valid grad = 2.0)
    assert w.grad.item() == 2.0
    
    # 2. Enter standard_mode context manager (SHA disabled)
    with torch.standard_mode():
        w2 = torch.Tensor([2.0], requires_grad=True)
        torch.Tensor._grad_history[id(w2)] = (np.array([2.0]).shape, np.array([2.0]))
        w2.grad = torch.Tensor([float('nan')])
        
        # Accessing grad should NOT heal the NaN
        assert np.isnan(w2.grad.item())
        
    # 3. Outside context manager (SHA enabled)
    w3 = torch.Tensor([2.0], requires_grad=True)
    torch.Tensor._grad_history[id(w3)] = (np.array([2.0]).shape, np.array([2.0]))
    w3.grad = torch.Tensor([float('nan')])
    
    # Accessing grad should trigger self-healing
    assert w3.grad.item() == 2.0
