import torch_candle as torch
import numpy as np

def test_auto_device_alignment():
    x = torch.Tensor([1.0, 2.0], device="cpu")
    y = torch.Tensor([3.0, 4.0], device="cpu")
    
    # Verify standard addition works perfectly
    res = x + y
    assert np.allclose(res.numpy(), np.array([4.0, 6.0]))
    
    # Simulate a device mismatch (CPU tensor and GPU tensor)
    # The Auto-Device Alignment Engine automatically converts other to self's device on-the-fly!
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x_dev = x.to(target_device)
    y_cpu = torch.Tensor([3.0, 4.0], device="cpu")
    
    # Addition with mixed devices: x_dev (GPU/CPU) + y_cpu (CPU)
    # This would crash in standard PyTorch, but Torch-Candle aligns it!
    res_aligned = x_dev + y_cpu
    
    assert res_aligned.device == target_device
    assert np.allclose(res_aligned.numpy(), np.array([4.0, 6.0]))
