import torch_candle as torch
import torch_candle.nn as nn
import torch_candle.amp as amp
import numpy as np

def test_amp_autocast_context():
    # Verify default state
    assert torch.Tensor._amp_enabled == False
    
    # Enter autocast context
    with amp.autocast(device_type="cuda", dtype="float16"):
        assert torch.Tensor._amp_enabled == True
        assert torch.Tensor._amp_dtype == "float16"
        
        # Nested autocast with different precision
        with amp.autocast(device_type="cuda", dtype="bfloat16"):
            assert torch.Tensor._amp_enabled == True
            assert torch.Tensor._amp_dtype == "bfloat16"
            
        # Reverts to outer state
        assert torch.Tensor._amp_dtype == "float16"
        
    # Reverts to default state
    assert torch.Tensor._amp_enabled == False

def test_amp_grad_scaler():
    scaler = amp.GradScaler(enabled=True)
    assert scaler.is_enabled() == True
    assert scaler.get_scale() == 65536.0
    
    x = torch.Tensor([1.0, 2.0])
    scaled_x = scaler.scale(x)
    assert np.allclose(scaled_x.numpy(), np.array([65536.0, 131072.0]))
