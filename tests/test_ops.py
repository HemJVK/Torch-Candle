import torch_candle as torch
import numpy as np
import pytest

def test_tensor_ops():
    x = torch.Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = torch.Tensor([[5, 6], [7, 8]], requires_grad=True)
    
    # Addition
    z = x + y
    assert np.allclose(z.numpy(), [[6, 8], [10, 12]])
    assert z.requires_grad
    
    # Subtraction
    z = x - y
    assert np.allclose(z.numpy(), [[-4, -4], [-4, -4]])
    
    # Multiplication
    z = x * y
    assert np.allclose(z.numpy(), [[5, 12], [21, 32]])
    
    # Matmul
    z = x.matmul(y)
    # [1*5+2*7, 1*6+2*8] = [19, 22]
    # [3*5+4*7, 3*6+4*8] = [43, 50]
    assert np.allclose(z.numpy(), [[19, 22], [43, 50]])
    
    # Sum
    s = x.sum()
    assert np.allclose(s.numpy(), 10.0)
    
    # Mean
    m = x.mean()
    assert np.allclose(m.numpy(), 2.5)

def test_autograd_ops():
    x = torch.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    
    # Test Transpose
    z = x.t()
    z.sum().backward()
    assert x.grad is not None
    assert np.allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]])
    
    x.grad = None
    # Test Pow
    z = x ** 2
    z.sum().backward()
    # d(x^2)/dx = 2x
    assert np.allclose(x.grad.numpy(), [[2.0, 4.0], [6.0, 8.0]])
    
    x.grad = None
    # Test Slice
    z = x[0, 0]
    z.backward()
    # Gradient should only be at [0, 0]
    expected_grad = [[1.0, 0.0], [0.0, 0.0]]
    assert np.allclose(x.grad.numpy(), expected_grad)

def test_reshape_view():
    x = torch.Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    z = x.view(2, 3)
    assert z.shape == (2, 3)
    z.sum().backward()
    assert np.allclose(x.grad.numpy(), [1, 1, 1, 1, 1, 1])
