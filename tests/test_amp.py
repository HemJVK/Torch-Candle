import pytest
import numpy as np
import torch_candle as torch
from torch_candle import amp

def test_autocast_context():
    # Test that autocast acts as a valid context manager and doesn't crash
    t = torch.tensor([1.0, 2.0])
    with amp.autocast(device_type='cuda', dtype=None):
        out = t * 2.0
    assert np.allclose(out.numpy(), [2.0, 4.0])

def test_autocast_decorator():
    @amp.autocast()
    def my_forward(x):
        return x + 1.0

    t = torch.tensor([1.0])
    res = my_forward(t)
    assert res.item() == 2.0

def test_gradscaler():
    # Test GradScaler stub methods (no-op in candle, but API should exist)
    scaler = amp.GradScaler()
    
    # Scale test
    loss = torch.tensor([1.0], requires_grad=True)
    scaled_loss = scaler.scale(loss)
    # The default stub might just return the loss identically or scaled float
    # We just ensure it doesn't crash
    assert scaled_loss is not None
    
    # Optimizer step stub
    class MockOptimizer:
        def step(self):
            self.stepped = True

    opt = MockOptimizer()
    scaler.step(opt)
    assert opt.stepped

    scaler.update()
    
    # State dict
    sd = scaler.state_dict()
    assert 'scale' in sd
    scaler.load_state_dict(sd)
