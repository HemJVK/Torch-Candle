import torch_candle as torch
import numpy as np
import pytest

def test_no_sha_context_manager():
    # Store initial state
    initial_state = torch.Tensor.enable_sha
    
    with torch.no_sha():
        # Inside the context manager, SHA should be disabled
        assert not torch.Tensor.enable_sha
        
    # Outside the context manager, it should be restored
    assert torch.Tensor.enable_sha == initial_state

def test_has_cudnn_and_nccl():
    import torch_candle_backend as _kernels
    assert isinstance(_kernels.has_cudnn(), bool)
    assert isinstance(_kernels.has_nccl(), bool)

def test_backward_with_no_sha():
    with torch.no_sha():
        w = torch.Tensor([2.0], requires_grad=True)
        loss = w * 3.0
        loss.backward()
        assert w.grad.item() == 3.0
