import torch_candle as torch
import torch_candle.nn as nn
import torch_candle.nn.functional as F
import numpy as np

def test_scaled_dot_product_attention():
    # Construct Q, K, V: (B, H, S, D) = (2, 2, 8, 16)
    q = torch.randn(2, 2, 8, 16)
    k = torch.randn(2, 2, 8, 16)
    v = torch.randn(2, 2, 8, 16)
    
    # Standard SDPA
    out = F.scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 2, 8, 16)
    
    # Causal SDPA
    out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert out_causal.shape == (2, 2, 8, 16)
    
    # Validate result is mathematically sound
    # Attention weights sum to 1.0 along the key sequence dimension
    d_k = 16.0
    scale = 1.0 / (d_k ** 0.5)
    scores = q.matmul(k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    expected_out = attn.matmul(v)
    assert np.allclose(out.numpy(), expected_out.numpy(), atol=1e-5)

def test_torch_compile():
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 2)
            
        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
            
    model = SimpleMLP()
    compiled_model = torch.compile(model)
    
    x = torch.randn(3, 4)
    
    # First forward pass (compiling / tracing)
    out1 = compiled_model(x)
    assert out1.shape == (3, 2)
    assert compiled_model.is_compiled == True
    
    # Subsequent forward pass (fast JIT path execution)
    out2 = compiled_model(x)
    assert out2.shape == (3, 2)
    
    # Both eager and JIT outputs should match perfectly
    assert np.allclose(out1.numpy(), out2.numpy(), atol=1e-6)
