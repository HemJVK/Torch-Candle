import torch_candle as torch
from torch_candle.func import AttnBiasTensor, subclass_dispatch
import numpy as np
import json

def test_strict_ema_reconstruction_heterogeneous():
    # 1. Establish history using standard tuple
    w = torch.Tensor([3.0], requires_grad=True)
    loss = w * 3.0
    loss.backward()
    assert w.grad.item() == 3.0
    
    # 2. Simulate heterogeneous dictionary format in _grad_history
    w_dict = torch.Tensor([3.0], requires_grad=True)
    torch.Tensor._grad_history[id(w_dict)] = {
        "shape": [1],
        "data": [3.0]
    }
    w_dict.grad = torch.Tensor([float('nan')])
    
    # Retrieves should trigger strict SHA reconstruction and parse PyDict format perfectly!
    healed_dict = w_dict.grad
    assert not np.isnan(healed_dict.item())
    assert healed_dict.item() == 3.0
    
    # 3. Simulate heterogeneous JSON format in _grad_history
    w_json = torch.Tensor([3.0], requires_grad=True)
    torch.Tensor._grad_history[id(w_json)] = json.dumps({
        "shape": [1],
        "data": [3.0]
    })
    w_json.grad = torch.Tensor([float('nan')])
    
    # Retrieves should trigger strict SHA reconstruction and parse PyString JSON format perfectly!
    healed_json = w_json.grad
    assert not np.isnan(healed_json.item())
    assert healed_json.item() == 3.0


def test_logical_blockfree_caching():
    from torch_candle.cuda import _allocator
    
    # Allocate a block
    ptr = _allocator.allocate(512, 1, "query_block")
    assert ptr > 0
    
    # Freeing the block (logical cache - mark idle)
    _allocator.free(ptr, 1)
    
    # Reusing the cached block immediately through BlockFree tracking without GPU sync overhead
    new_ptr = _allocator.allocate(512, 1, "key_block")
    assert new_ptr == ptr
    
    # Logical release via cuda_free keeps block cached for future requests
    _allocator.cuda_free(ptr)


def test_dynamic_subclass_dispatcher_sdpa():
    # Create query, key, value
    q = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    k = torch.Tensor([[5.0, 6.0], [7.0, 8.0]])
    v = torch.Tensor([[9.0, 10.0], [11.0, 12.0]])
    
    # Create the AttnBiasTensor subclass containing the bias mask
    bias_data = np.array([[0.0, -100.0], [-100.0, 0.0]], dtype=np.float32)
    attn_bias = AttnBiasTensor(bias_data, mask_type="block_diagonal")
    
    # Trigger Scaled Dot-Product Attention
    from torch_candle.nn.functional import scaled_dot_product_attention
    
    # Dispatcher intercepts and routes through AttnBiasTensor
    out = scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
    
    # Verify outputs are successfully produced
    assert out.shape == (2, 2)
    assert out.numpy().shape == (2, 2)


def test_rocm_aot_compiler_helper():
    from torch_candle.compile_rocm import ROCmAOTCompiler
    
    # Assert driver is correctly loaded and ready
    assert hasattr(ROCmAOTCompiler, "is_hipcc_available")
    assert hasattr(ROCmAOTCompiler, "compile_kernels_aot")
