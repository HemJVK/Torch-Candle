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


def test_new_activations():
    x = torch.tensor([-2.0, 0.0, 2.0])
    
    # Tanh
    tanh = nn.Tanh()
    assert np.allclose(tanh(x).numpy(), np.tanh([-2.0, 0.0, 2.0]), atol=1e-5)
    
    # LeakyReLU
    leaky = nn.LeakyReLU(negative_slope=0.1)
    assert np.allclose(leaky(x).numpy(), np.array([-0.2, 0.0, 2.0]), atol=1e-5)
    
    # ELU
    elu = nn.ELU(alpha=1.0)
    expected_elu = np.array([1.0 * (np.exp(-2.0) - 1.0), 0.0, 2.0], dtype=np.float32)
    assert np.allclose(elu(x).numpy(), expected_elu, atol=1e-5)
    
    # SELU
    selu = nn.SELU()
    _alpha = 1.6732632423543772
    _scale = 1.0507009873554805
    expected_selu = np.array([_alpha * (np.exp(-2.0) - 1.0), 0.0, 2.0], dtype=np.float32) * _scale
    assert np.allclose(selu(x).numpy(), expected_selu, atol=1e-5)
    
    # PReLU
    prelu = nn.PReLU(num_parameters=1, init=0.5)
    assert np.allclose(prelu(x).numpy(), np.array([-1.0, 0.0, 2.0], dtype=np.float32), atol=1e-5)
