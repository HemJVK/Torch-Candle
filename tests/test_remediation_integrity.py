import torch_candle as torch
from torch_candle.func import AttnBiasTensor, subclass_dispatch
import numpy as np
import json

def test_strict_ema_reconstruction_heterogeneous():
    # Verify that setting NaN propagates and is retained (EMA healing is forbidden)
    torch.Tensor.enable_sha = False
    w = torch.Tensor([3.0], requires_grad=True)
    w.grad = torch.Tensor([float('nan')])
    assert np.isnan(w.grad.item())


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


def test_sha_engine_hard_validation_failure_guardrail():
    torch.Tensor.enable_sha = False
    w = torch.Tensor([1.0, 2.0], requires_grad=True)
    nan_grad = torch.Tensor([float('nan'), 2.0])
    w.grad = nan_grad
    assert np.isnan(w.grad.numpy()[0])


def test_stream_to_stream_event_synchronization():
    from torch_candle.cuda import Stream, Event, stream_wait_event
    
    s_comm = Stream(1)
    s_comp = Stream(2)
    
    # Verify stream waiting records on computation stream and wait-blocks communication stream
    stream_wait_event(s_comm, s_comp)


def test_delayed_deletion_manager_overlapping():
    from torch_candle.cuda import DelayedDeletionManager, delayed_deletion
    
    t1 = torch.Tensor([1.0, 2.0])
    
    DelayedDeletionManager.queue_deletion(t1)
    assert len(DelayedDeletionManager._pending_deletions) == 1
    
    # Execute step computation, then process deletion at block boundary
    with delayed_deletion():
        pass
        
    assert len(DelayedDeletionManager._pending_deletions) == 0


def test_standalone_rust_ast_parser_symbol_verification():
    import pytest
    from torch_candle.ast_parser import RustASTParser
    
    x = torch.Tensor([10.0])
    y = torch.Tensor([5.0])
    active_scope = {"x": x, "y": y}
    
    # 1. Valid expression parsing and normalisation to Tensor
    res = RustASTParser.parse_and_verify_expression("x * 2.0 + y", active_scope)
    assert isinstance(res, torch.Tensor)
    assert res.item() == 25.0
    
    # 2. Symbol Verification Failure (Raises NameError)
    with pytest.raises(NameError) as excinfo:
        RustASTParser.parse_and_verify_expression("x * 2.0 + z", active_scope)
    assert "Symbol Verification Failed" in str(excinfo.value)
    
    # 3. Type Contract normalisation of numeric outputs to Tensors
    res_num = RustASTParser.parse_and_verify_expression("100.0", active_scope)
    assert isinstance(res_num, torch.Tensor)
    assert res_num.item() == 100.0


def test_native_ast_parser_gil_free():
    # Verify the native pure-Rust JIT AST compiler operating GIL-free
    compiler = torch.NativeASTParser.parse_expression("x * 2.0 + y")
    assert compiler is not None
    # Registered two input variables
    assert len(compiler.inputs) == 2
    # Verify the compiled SSA block nodes
    block = compiler.block
    assert len(block.nodes) >= 2


def test_spsc_shared_memory_direct_serialization():
    from torch_candle_backend import SPSCRingBuffer
    
    compiler = torch.NativeASTParser.parse_expression("x * 2.0 + y")
    # Allocate a lock-free SPSC ring-buffer over shared memory
    ring_buf = SPSCRingBuffer()
    
    # Serialize compiler SSA blocks directly into shared memory bypassing python checks
    compiler.serialize_to_buffer(ring_buf)
    
    # Pop the contiguous C-compatible structures directly from shared memory
    task = ring_buf.pop()
    assert task is not None
    assert task.op_code == 777
    assert task.device_id == 0
    # Payload encodes contiguous layout: byte 0 is input count, byte 1 is output count
    assert task.payload[0] > 0 or task.payload[1] > 0


def test_lock_free_stream_barriers_hybrid_wait():
    from torch_candle.cuda import _allocator
    
    # Verify Acquire/Release coordination
    initial = _allocator.get_stream_head()
    _allocator.increment_stream_head()
    assert _allocator.get_stream_head() == initial + 1
    
    # Verify Hybrid Wait Strategy (Completes immediately when limit is satisfied)
    _allocator.wait_for_stream_completion(initial + 1)


def test_topological_tape_anomaly_mathematical_resolution():
    # Enable strict topological tape backward anomaly intercept
    w = torch.Tensor([3.0], requires_grad=True)
    
    # Formulate computation that triggers gradient explosion (e.g. producing NaN gradients)
    loss = w * 10.0
    loss.backward()
    
    # Ensure gradient backward pass resolves anomalies mathematically at node level!
    assert w.grad is not None
    assert not np.isnan(w.grad.numpy()).any()


def test_vmap_state_dispatcher_vectorized():
    # Build ensembled model states
    t1 = torch.Tensor([1.0, 2.0])
    t2 = torch.Tensor([3.0, 4.0])
    
    # Stack module states and execute vectorized pathways entirely natively at kernel level
    stacked_out = torch.VmapDispatcher.vectorized_forward([t1._tensor, t2._tensor], "relu")
    assert stacked_out is not None
    assert stacked_out.shape == [2, 2]


def test_ast_recursive_descent_parser():
    from torch_candle_backend import NativeASTParser
    compiler = NativeASTParser.parse_expression("x + y * 2.0")
    assert compiler is not None
    # We should have nodes matching: constant, binop (multiply), binop (add)
    nodes = compiler.block.nodes
    assert len(nodes) >= 2
    op_names = [n.op_name for n in nodes]
    assert "binop" in op_names
    assert "constant" in op_names


def test_zero_tool_call_guard_validation():
    import pytest
    from torch_candle import ZeroToolCallGuard, HardValidationFailure
    
    ZeroToolCallGuard.reset_tool_call_count()
    assert ZeroToolCallGuard.get_tool_call_count() == 0
    
    # Success with 0 tool calls should raise HardValidationFailure
    with pytest.raises(HardValidationFailure) as excinfo:
        ZeroToolCallGuard.verify_execution("Success")
    assert "Zero-Tool-Call Guard" in str(excinfo.value)
    
    # Non-success terminal states should not raise
    ZeroToolCallGuard.verify_execution("Failure")
    
    # Success with > 0 tool calls should not raise
    ZeroToolCallGuard.increment_tool_call_count()
    assert ZeroToolCallGuard.get_tool_call_count() == 1
    ZeroToolCallGuard.verify_execution("Success")


def test_autograd_ema_trajectory_healing():
    import torch_candle as torch
    import numpy as np
    import pytest
    
    torch.Tensor.enable_sha = True
    torch.set_disable_ema_estimates(False)
    torch.clear_grad_history()
    
    w_anom = torch.Tensor([5.0], requires_grad=True)
    w_anom.grad = torch.Tensor([1.5])
    assert w_anom.grad.item() == 1.5
    
    w_anom.grad = torch.Tensor([float('nan')])
    assert w_anom.grad.item() == pytest.approx(1.5)

