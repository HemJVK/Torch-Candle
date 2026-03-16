import pytest
import numpy as np
import torch_candle as torch
from torch_candle import nn
import math

def test_gelu():
    x = torch.tensor([-1.0, 0.0, 1.0])
    gelu = nn.GELU()
    out = gelu(x)
    
    # Expected GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    # GELU(0) = 0
    # GELU(1) = 1 * 0.5 * (1 + 0.8427) = ~0.8413
    # GELU(-1) = -1 * 0.5 * (1 - 0.8427) = ~-0.1587
    
    expected = np.array([-0.158655, 0.0, 0.841345], dtype=np.float32)
    assert np.allclose(out.numpy(), expected, atol=1e-4)

def test_softmax():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    softmax = nn.Softmax(dim=1)
    out = softmax(x)
    
    expected = np.exp([1.0, 2.0, 3.0]) / np.sum(np.exp([1.0, 2.0, 3.0]))
    assert np.allclose(out.numpy(), expected, atol=1e-5)
