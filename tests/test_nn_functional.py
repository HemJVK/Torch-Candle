import pytest
import numpy as np
import torch_candle as torch
import torch_candle.nn.functional as F

def test_gelu():
    t = torch.tensor([-1.0, 0.0, 1.0])
    out = F.gelu(t)
    # GELU(-1) ~ -0.1587, GELU(0) = 0, GELU(1) ~ 0.8413
    expected = np.array([-0.158655, 0.0, 0.841345])
    assert np.allclose(out.numpy(), expected, atol=1e-4)

def test_silu():
    t = torch.tensor([-1.0, 0.0, 1.0])
    out = F.silu(t)
    # SiLU(x) = x * sigmoid(x)
    # SiLU(-1) = -1 * 0.2689 = -0.2689
    # SiLU(1) = 1 * 0.7311 = 0.7311
    expected = np.array([-0.268941, 0.0, 0.731059])
    assert np.allclose(out.numpy(), expected, atol=1e-4)

def test_batch_norm():
    input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]) # (1, 1, 2, 2)
    running_mean = torch.zeros(1)
    running_var = torch.ones(1)
    weight = torch.ones(1)
    bias = torch.zeros(1)

    # training=True adapts running stats
    out = F.batch_norm(input, running_mean, running_var, weight, bias, training=True)
    
    mean = 2.5
    var = 1.25 # (1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4 = (2.25 + 0.25 + 0.25 + 2.25)/4 = 5/4 = 1.25
    expected_out = (np.array([1.0, 2.0, 3.0, 4.0]) - mean) / np.sqrt(var + 1e-5)
    assert np.allclose(out.numpy().flatten(), expected_out, atol=1e-4)
    # Check running stats update
    assert running_mean.item() != 0.0
    assert running_var.item() != 1.0

def test_layer_norm():
    input = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = F.layer_norm(input, [4])
    mean = 2.5
    var = 1.25
    expected = (np.array([1.0, 2.0, 3.0, 4.0]) - mean) / np.sqrt(var + 1e-5)
    assert np.allclose(out.numpy().flatten(), expected, atol=1e-4)

def test_binary_cross_entropy():
    input = torch.tensor([0.2, 0.8])
    target = torch.tensor([0.0, 1.0])
    loss = F.binary_cross_entropy(input, target)
    # - (0 * log(0.2) + 1 * log(0.8) + 1 * log(0.8) + 0 * log(0.2)) / 2 
    # = - (2 * log(0.8)) / 2  = -log(0.8) = 0.2231
    assert abs(loss.item() - 0.223143) < 1e-4

def test_huber_loss():
    input = torch.tensor([0.5, 2.0])
    target = torch.tensor([0.0, 0.0])
    loss = F.huber_loss(input, target, delta=1.0)
    # diff = [0.5, 2.0]
    # for 0.5 < 1.0 -> 0.5 * 0.5^2 = 0.125
    # for 2.0 >= 1.0 -> 2.0 - 0.5 = 1.5
    # mean = (0.125 + 1.5) / 2 = 1.625 / 2 = 0.8125
    assert abs(loss.item() - 0.8125) < 1e-4

def test_pad():
    input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded = F.pad(input, [1, 1, 1, 1], mode='constant', value=0.0)
    assert padded.shape == (4, 4)
    assert padded[0, 0].item() == 0.0
    assert padded[1, 1].item() == 1.0

def test_interpolate_2d():
    input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    out = F.interpolate(input, scale_factor=2.0)
    assert out.shape == (1, 1, 4, 4)
    # Check simple reproduction
    assert out[0, 0, 0, 0].item() == 1.0
    assert out[0, 0, -1, -1].item() == 4.0
