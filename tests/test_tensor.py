import torch_candle as torch
import pytest
import numpy as np

def test_tensor_init():
    t = torch.Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.numpy().tolist() == [1.0, 2.0, 3.0]

def test_tensor_add():
    t1 = torch.Tensor([1.0, 2.0])
    t2 = torch.Tensor([3.0, 4.0])
    t3 = t1 + t2
    assert t3.numpy().tolist() == [4.0, 6.0]

def test_tensor_mul():
    t1 = torch.Tensor([2.0, 3.0])
    t2 = torch.Tensor([4.0, 5.0])
    t3 = t1 * t2
    assert t3.numpy().tolist() == [8.0, 15.0]

def test_broadcasting_add():
    t1 = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    t2 = torch.Tensor([1.0, 1.0])
    t3 = t1 + t2
    expected = [[2.0, 3.0], [4.0, 5.0]]
    assert t3.numpy().tolist() == expected

def test_relu():
    t = torch.Tensor([-1.0, 0.0, 1.0])
    r = t.relu()
    assert r.numpy().tolist() == [0.0, 0.0, 1.0]

def test_tensor_sub():
    t1 = torch.Tensor([5.0, 7.0])
    t2 = torch.Tensor([2.0, 3.0])
    t3 = t1 - t2
    assert t3.numpy().tolist() == [3.0, 4.0]

def test_tensor_div():
    t1 = torch.Tensor([10.0, 20.0])
    t2 = torch.Tensor([2.0, 4.0])
    t3 = t1 / t2
    assert t3.numpy().tolist() == [5.0, 5.0]

def test_scalar_arithmetic():
    t = torch.Tensor([1.0, 2.0])
    assert (t + 1.0).numpy().tolist() == [2.0, 3.0]
    assert (t * 2.0).numpy().tolist() == [2.0, 4.0]
    assert (t - 1.0).numpy().tolist() == [0.0, 1.0]
    assert (t / 2.0).numpy().tolist() == [0.5, 1.0]

def test_radd_rmul():
    t = torch.Tensor([1.0, 2.0])
    assert (1.0 + t).numpy().tolist() == [2.0, 3.0]
    assert (2.0 * t).numpy().tolist() == [2.0, 4.0]

def test_neg():
    t = torch.Tensor([1.0, -2.0])
    assert (-t).numpy().tolist() == [-1.0, 2.0]

def test_shapes():
    t = torch.randn((2, 3, 4))
    assert t.shape == (2, 3, 4)
    assert t.size(0) == 2
    assert t.size(1) == 3
    assert t.size(2) == 4

def test_reshape_view():
    t = torch.zeros((4, 4))
    t2 = t.view(2, 8)
    assert t2.shape == (2, 8)
    t3 = t.reshape(16)
    assert t3.shape == (16,)

def test_mean_sum():
    t = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.sum().item() == 10.0
    assert t.mean().item() == 2.5
    assert t.sum(dim=0).numpy().tolist() == [4.0, 6.0]

def test_complex_indexing():
    t = torch.randn(4, 4, 4)
    # Advanced indexing
    sub = t[1:3, :, 0]
    assert sub.shape == (2, 4)
    
    # Ellipsis-like (simple tuple)
    sub2 = t[(slice(None), 1, slice(None))]
    assert sub2.shape == (4, 4)

def test_setitem_simple():
    t = torch.zeros(5)
    t[0] = 1.0
    t[2:4] = torch.Tensor([2.0, 3.0])
    assert t.numpy().tolist() == [1.0, 0.0, 2.0, 3.0, 0.0]

def test_setitem_advanced():
    t = torch.zeros(3, 3)
    t[1, 1] = 5.0
    t[0, :] = 1.0
    expected = [
        [1.0, 1.0, 1.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
    assert t.numpy().tolist() == expected

if __name__ == "__main__":
    pytest.main([__file__])
