import torch_candle as torch
import numpy as np

def test_inplace_arithmetic():
    x = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.Tensor([[1.0, 1.0], [1.0, 1.0]])
    
    # Test add_
    x.add_(y)
    assert np.allclose(x.numpy(), np.array([[2.0, 3.0], [4.0, 5.0]]))
    
    # Test sub_
    x.sub_(y)
    assert np.allclose(x.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    # Test mul_
    x.mul_(y * 2.0)
    assert np.allclose(x.numpy(), np.array([[2.0, 4.0], [6.0, 8.0]]))
    
    # Test div_
    x.div_(y * 2.0)
    assert np.allclose(x.numpy(), np.array([[1.0, 2.0], [3.0, 4.0]]))

def test_inplace_operators():
    x = torch.Tensor([2.0])
    
    # Test +=
    x += 3.0
    assert np.allclose(x.numpy(), np.array([5.0]))
    
    # Test -=
    x -= 1.0
    assert np.allclose(x.numpy(), np.array([4.0]))
    
    # Test *=
    x *= 2.0
    assert np.allclose(x.numpy(), np.array([8.0]))
    
    # Test /=
    x /= 4.0
    assert np.allclose(x.numpy(), np.array([2.0]))

def test_inplace_activations():
    x = torch.Tensor([-2.0, 3.0])
    
    # Test relu_
    x.relu_()
    assert np.allclose(x.numpy(), np.array([0.0, 3.0]))
    
    # Test sigmoid_
    y = torch.Tensor([0.0])
    y.sigmoid_()
    assert np.allclose(y.numpy(), np.array([0.5]))
