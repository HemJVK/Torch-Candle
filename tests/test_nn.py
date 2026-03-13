import torch_candle as torch
import torch_candle.nn as nn
import pytest
import numpy as np

def test_linear_forward():
    lin = nn.Linear(3, 2)
    x = torch.randn(1, 3)
    out = lin(x)
    assert out.shape == (1, 2)

def test_mseloss():
    criterion = nn.MSELoss()
    input = torch.Tensor([1.0, 2.0], requires_grad=True)
    target = torch.Tensor([0.0, 0.0])
    loss = criterion(input, target)
    
    # loss = (1.0^2 + 2.0^2)/2 = (1+4)/2 = 2.5
    assert loss.item() == 2.5
    
    loss.backward()
    # dL/di = 2*(i-t)/N = (i-t) = [1.0, 2.0]
    assert input.grad.numpy().tolist() == [1.0, 2.0]

def test_training_step():
    # Simple linear regression: y = 2x
    model = nn.Linear(1, 1, bias=False)
    # Force weight to something known for predictability
    model.weight.data[0, 0] = 5.0
    
    x = torch.Tensor([[1.0]])
    y = torch.Tensor([[2.0]])
    
    criterion = nn.MSELoss()
    
    # Forward
    pred = model(x)
    print(f"DEBUG: pred.grad_fn={pred.grad_fn}")
    loss = criterion(pred, y)
    print(f"DEBUG: loss.grad_fn={loss.grad_fn}")
    if loss.grad_fn:
        print(f"DEBUG: loss.grad_fn.inputs={[type(i) for i in loss.grad_fn.inputs]}")
        print(f"DEBUG: loss.grad_fn.inputs requires_grad={[i.requires_grad for i in loss.grad_fn.inputs if isinstance(i, torch.Tensor)]}")
    
    # Backward
    loss.backward()
    
    # Grad check: 
    # pred = w*x = 5*1 = 5
    # loss = (5-2)^2 = 9 (no reduction on scalar)
    # dL/dw = dL/dpred * dpred/dw = 2*(pred-y) * x = 2*(5-2) * 1 = 6
    assert model.weight.grad.item() == 6.0

if __name__ == "__main__":
    pytest.main([__file__])
