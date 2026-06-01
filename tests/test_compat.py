import sys
import pytest

def test_pytorch_compat_layer():
    # Make sure 'torch' is not in active system modules initially
    if 'torch' in sys.modules and sys.modules['torch'].__name__ == 'torch':
        sys.modules['real_torch'] = sys.modules['torch']
    if 'torch' in sys.modules:
        del sys.modules['torch']
        
    import torch_candle
    torch_candle.enable_torch_compat()
    
    # Standard PyTorch style imports
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Verify module identities
    assert torch == torch_candle
    assert nn == torch_candle.nn
    assert F == torch_candle.nn.functional
    
    # Verify functionality via standard PyTorch names
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
    y = F.relu(x)
    
    assert y.shape == (2, 2)
    assert y.numpy()[0, 0] == 0.0
    assert y.numpy()[0, 1] == 2.0
