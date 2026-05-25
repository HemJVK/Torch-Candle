import pytest
import numpy as np
import torch_candle as torch
from torch_candle import nn
from torch_candle.autograd import Function

# ──────────────────────────────────────────────────────────────
# 1. Custom Autograd Function Test
# ──────────────────────────────────────────────────────────────

class MultiplyAdd(Function):
    @staticmethod
    def forward(ctx, x, y, z):
        ctx.save_for_backward(x, y, z)
        return x * y + z

    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        return grad_output * y, grad_output * x, grad_output

def test_custom_autograd_function():
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    z = torch.tensor([4.0], requires_grad=True)
    
    # Forward: 2 * 3 + 4 = 10
    out = MultiplyAdd.apply(x, y, z)
    assert out.item() == 10.0
    
    # Backward
    out.backward()
    
    # grads:
    # dL/dx = y = 3
    # dL/dy = x = 2
    # dL/dz = 1
    assert x.grad.item() == 3.0
    assert y.grad.item() == 2.0
    assert z.grad.item() == 1.0


# ──────────────────────────────────────────────────────────────
# 2. Pickling / Serialization Test
# ──────────────────────────────────────────────────────────────

def test_serialization(tmp_path):
    filepath = tmp_path / "model.pt"
    
    # Create simple model
    model = nn.Sequential(
        nn.Linear(2, 2, bias=False)
    )
    # Force weight
    model[0].weight.data[0, 0] = 42.0
    
    # Save model state dict
    state = model.state_dict()
    torch.save(state, filepath)
    
    # Load back
    loaded_state = torch.load(filepath)
    
    # Verify loaded value
    assert loaded_state["0.weight"].numpy()[0, 0] == 42.0


# ──────────────────────────────────────────────────────────────
# 3. Transformer Test
# ──────────────────────────────────────────────────────────────

def test_transformer():
    # d_model=8, nhead=2
    mha = nn.MultiheadAttention(embed_dim=8, num_heads=2)
    
    # seq_len=4, batch=2, embed=8
    x = torch.randn(4, 2, 8)
    
    # Forward pass
    out, weights = mha(x, x, x)
    assert out.shape == (4, 2, 8)
    
    # Transformer Encoder Layer
    enc_layer = nn.TransformerEncoderLayer(d_model=8, nhead=2)
    enc_out = enc_layer(x)
    assert enc_out.shape == (4, 2, 8)
    
    # Transformer Encoder
    encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
    encoder_out = encoder(x)
    assert encoder_out.shape == (4, 2, 8)

if __name__ == "__main__":
    pytest.main([__file__])
