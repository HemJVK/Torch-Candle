import torch_candle as torch
import pytest

def test_backward_simple_add():
    x = torch.Tensor([2.0], requires_grad=True)
    y = torch.Tensor([3.0], requires_grad=True)
    z = x + y
    z.backward()
    
    assert x.grad.numpy().tolist() == [1.0]
    assert y.grad.numpy().tolist() == [1.0]

def test_backward_mul():
    x = torch.Tensor([2.0], requires_grad=True)
    y = torch.Tensor([3.0], requires_grad=True)
    z = x * y
    z.backward()
    
    # dz/dx = y = 3
    # dz/dy = x = 2
    assert x.grad.numpy().tolist() == [3.0]
    assert y.grad.numpy().tolist() == [2.0]

def test_backward_chain():
    x = torch.Tensor([2.0], requires_grad=True)
    y = torch.Tensor([3.0], requires_grad=True)
    # z = x * y + x
    z = (x * y) + x
    z.backward()
    
    # dz/dx = y + 1 = 3 + 1 = 4
    # dz/dy = x = 2
    assert x.grad.numpy().tolist() == [4.0]
    assert y.grad.numpy().tolist() == [2.0]

if __name__ == "__main__":
    pytest.main([__file__])
