import pytest
import math
import numpy as np
import torch_candle as torch
from torch_candle import optim

def test_sgd():
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    w.grad = torch.tensor([0.1, -0.1])
    optimizer = optim.SGD([w], lr=0.1, momentum=0.9, weight_decay=0.01)
    optimizer.step()
    # (weight_decay) -> g = g + wd * w = [0.1 + 0.01, -0.1 + 0.02] = [0.11, -0.08]
    # (momentum) -> buf = g = [0.11, -0.08]
    # w = w - lr * buf = [1.0 - 0.011, 2.0 - (-0.008)] = [0.989, 2.008]
    assert np.allclose(w.numpy(), [0.989, 2.008], atol=1e-4)

def test_adam():
    w = torch.tensor([1.0], requires_grad=True)
    w.grad = torch.tensor([0.1])
    optimizer = optim.Adam([w], lr=0.1)
    optimizer.step()
    # Simple check that step doesn't crash and slightly decreases w
    assert w.item() < 1.0

def test_adagrad():
    w = torch.tensor([1.0], requires_grad=True)
    w.grad = torch.tensor([0.1])
    optimizer = optim.Adagrad([w], lr=0.1)
    optimizer.step()
    assert np.allclose(optimizer.state[w]['sum'], [0.01])
    assert w.item() < 1.0

def test_rmsprop():
    w = torch.tensor([1.0], requires_grad=True)
    w.grad = torch.tensor([0.1])
    optimizer = optim.RMSprop([w], lr=0.1)
    optimizer.step()
    assert w.item() < 1.0

def test_adamax():
    w = torch.tensor([1.0], requires_grad=True)
    w.grad = torch.tensor([0.1])
    optimizer = optim.Adamax([w], lr=0.1)
    optimizer.step()
    assert w.item() < 1.0

def test_nadam():
    w = torch.tensor([1.0], requires_grad=True)
    w.grad = torch.tensor([0.1])
    optimizer = optim.NAdam([w], lr=0.1)
    optimizer.step()
    assert w.item() < 1.0

def test_radam():
    w = torch.tensor([1.0], requires_grad=True)
    w.grad = torch.tensor([0.1])
    optimizer = optim.RAdam([w], lr=0.1)
    optimizer.step()
    assert w.item() < 1.0

def test_steplr():
    w = torch.tensor([1.0])
    optimizer = optim.SGD([w], lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    assert optimizer.param_groups[0]['lr'] == 0.1
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 0.1
    scheduler.step() # epoch 2 -> decay
    assert abs(optimizer.param_groups[0]['lr'] - 0.01) < 1e-6

def test_cosine_annealing():
    w = torch.tensor([1.0])
    optimizer = optim.SGD([w], lr=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    assert lr < 0.1 and lr > 0.0
