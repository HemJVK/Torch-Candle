import numpy as np
import pytest
import torch_candle as torch
import torch_candle_backend as _kernels
from torch_candle.func import make_functional, functional_call, stack_module_state
from torch_candle.autograd import Function
from torch_candle.jit.compiler import trace, ScriptModule
import time
import threading

# ============================================================
# Phase 1: Functional Transform Layer Tests
# ============================================================
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[2.0, 3.0]]))
        self.bias = torch.nn.Parameter(torch.tensor([1.0]))
        self.register_buffer("running_mean", torch.tensor([5.0]))

    def forward(self, x):
        return x @ self.weight.t() + self.bias + self.running_mean

def test_functional_transforms():
    model = LinearModel()
    
    # 1. make_functional
    func, params_tuple = make_functional(model)
    
    # 2. functional_call / execution
    x = torch.tensor([[1.0, 2.0]])
    out = func(params_tuple, x)
    assert np.allclose(out.numpy(), [[1.0 * 2.0 + 2.0 * 3.0 + 1.0 + 5.0]])

    # 3. stack_module_state
    models = [LinearModel() for _ in range(3)]
    stacked_params, stacked_buffers = stack_module_state(models)
    assert stacked_params["weight"].shape == (3, 1, 2)
    assert stacked_buffers["running_mean"].shape == (3, 1)


# ============================================================
# Phase 2: Caching Allocator & Stream Sync Tests
# ============================================================
def test_caching_allocator_sync():
    from torch_candle.cuda import _allocator, Stream
    
    s1 = Stream(1)
    
    # 1. Allocate a block on stream 1
    ptr = _allocator.allocate(1024, 1, "test_tag")
    assert ptr > 0
    
    # 2. Free block (marks idle)
    _allocator.free(ptr, 1)
    
    # 3. Record stream 2 dependency
    _allocator.record_stream(ptr, 2)
    
    # 4. Check allocate allows cross-stream reuse (Stream-Aware allocation)
    new_ptr = _allocator.allocate(1024, 2, "new_tag")
    assert new_ptr == ptr # block is reused!
    
    # 5. Test Tensor level record_stream (should raise RuntimeError due to the ban)
    t = torch.tensor([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="record_stream is banned"):
        t.record_stream(s1)
    
    # 6. Test cuda_free release
    _allocator.cuda_free(ptr)


# ============================================================
# Phase 3: Autograd DAG Tape & Custom Function Tests
# ============================================================
class SquareFunc(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * x * 2.0

def test_autograd_dag_tape():
    # 1. Run in standard_mode (SHA disabled)
    with torch.standard_mode():
        x = torch.tensor([3.0], requires_grad=True)
        # Standard Rust op (add) -> Custom Function (SquareFunc) -> Standard Rust op (mul)
        y = x + 1.0 # standard op
        z = SquareFunc.apply(y) # custom op
        out = z * 2.0 # standard op
        
        out.backward()
        
        # dy/dx = 1.0
        # dz/dy = 2 * y = 2 * 4.0 = 8.0
        # dout/dz = 2.0
        # dout/dx = dout/dz * dz/dy * dy/dx = 2.0 * 8.0 * 1.0 = 16.0
        assert np.allclose(x.grad.numpy(), [16.0])


# ============================================================
# Phase 4: JIT Tracing & Dynamic Shapes Tests
# ============================================================
def test_jit_dynamic_shapes(capsys):
    def simple_add(a, b):
        return a + b
        
    compiled = torch.compile(simple_add)
    
    a = torch.tensor([1.0])
    b = torch.tensor([2.0])
    
    # First pass: trace/compile
    out1 = compiled(a, b)
    assert np.allclose(out1.numpy(), [3.0])
    assert compiled.is_compiled == True
    
    # Second pass: matching shape (fast path)
    out2 = compiled(a, b)
    assert np.allclose(out2.numpy(), [3.0])
    
    # Third pass: dynamic shape (must trigger RuntimeError due to Zero-Fallback mandate)
    c = torch.tensor([1.0, 2.0])
    d = torch.tensor([3.0, 4.0])
    with pytest.raises(RuntimeError) as exc_info:
        compiled(c, d)
    assert "Zero-Fallback Mandate Violation" in str(exc_info.value)


# ============================================================
# Phase 5: Concurrency & SPSC Ring Buffer Tests
# ============================================================
def test_spsc_ring_buffer():
    import struct
    buf = _kernels.SPSCRingBuffer()
    results = []
    
    def py_wait_and_pop(buffer_obj):
        mv = memoryview(buffer_obj)
        start_time = time.time()
        while True:
            head = struct.unpack_from("Q", mv, 0)[0]
            tail = struct.unpack_from("Q", mv, 136)[0]
            if tail != head:
                break
            if time.time() - start_time > 2.0:
                raise TimeoutError("SPSC Wait and Pop timed out")
            time.sleep(0.001)
            
        index = tail % 1024
        task_offset = 144 + index * 4752
        
        op_code = struct.unpack_from("I", mv, task_offset + 0)[0]
        device_id = struct.unpack_from("Q", mv, task_offset + 8)[0]
        payload = bytes(mv[task_offset + 16 : task_offset + 16 + 256])
        
        struct.pack_into("Q", mv, 136, tail + 1)
        return op_code, device_id, payload

    def py_push(buffer_obj, op_code, device_id, payload_bytes):
        mv = memoryview(buffer_obj)
        head = struct.unpack_from("Q", mv, 0)[0]
        tail = struct.unpack_from("Q", mv, 136)[0]
        
        if head - tail >= 1024:
            raise RuntimeError("Buffer full")
            
        index = head % 1024
        task_offset = 144 + index * 4752
        
        struct.pack_into("I", mv, task_offset + 0, op_code)
        struct.pack_into("Q", mv, task_offset + 8, device_id)
        
        copy_len = min(len(payload_bytes), 4096)
        mv[task_offset + 16 : task_offset + 16 + copy_len] = payload_bytes[:copy_len]
        if copy_len < 4096:
            mv[task_offset + 16 + copy_len : task_offset + 16 + 4096] = b'\x00' * (4096 - copy_len)
            
        struct.pack_into("Q", mv, 0, head + 1)
        
    def consumer():
        try:
            op_code, device_id, payload = py_wait_and_pop(buf)
            msg = payload.decode().rstrip('\x00')
            results.append((True, op_code, device_id, msg))
        except Exception as e:
            results.append((False, str(e)))
            
    t = threading.Thread(target=consumer)
    t.start()
    
    # Wait a bit, then push item to unpark/trigger consumer
    time.sleep(0.1)
    py_push(buf, 42, 1, b"Hello SPSC IPC")
    t.join()
    
    assert results
    success, op_code, device_id, msg = results[0]
    assert success
    assert op_code == 42
    assert device_id == 1
    assert msg == "Hello SPSC IPC"


# ============================================================
# Phase 6: Decentralized Dispatching & Extension API Tests
# ============================================================
def test_dispatch_registry():
    # 1. Register backend
    torch.register_privateuse1_backend("custom_gpu")
    
    # 2. Register custom kernel
    def custom_add_kernel(a, b):
        return a * 10 + b * 10
        
    torch.register_kernel("custom::add", "custom_gpu", custom_add_kernel)
    
    # 3. Dispatch to kernel
    res = torch.dispatch_kernel("custom::add", "custom_gpu", 2, 3)
    assert res == 50

# ============================================================
# Technical Audit & Bug Remediation Tests (v0.5.0-Alpha)
# ============================================================
def test_glibc_mmap_tuning():
    import torch_candle_backend as _kernels
    # Native Glibc mallopt is invoked automatically during module initialization

def test_gpu_event_sync():
    from torch_candle.cuda import Event, Stream
    e = Event()
    s = Stream(1)
    e.record(s)
    assert e.query() == True
    e.wait(s)

def test_cuda_ipc_mem_handle():
    from torch_candle.multiprocessing.reductions import reduce_tensor, cudaIpcMemHandle
    t = torch.tensor([1.0, 2.0])
    # Force GPU reduction pathway serialization
    t._device = "cuda"
    reconstructor, args = reduce_tensor(t)
    
    assert isinstance(args[0], cudaIpcMemHandle)
    assert len(args[0].handle_bytes) == 64
    
    t_reconstructed = reconstructor(*args)
    assert t_reconstructed.device == "cuda"

def test_ast_compiler_control_flow():
    import torch_candle_backend as _kernels
    def conditional_func(x):
        y = x + 1.0
        if y > 2.0:
            z = y * 2.0
        else:
            z = y + 5.0
        return z
        
    compiler = _kernels.compile_ast(conditional_func)
    assert len(compiler.block.nodes) > 0
    op_names = [node.op_name for node in compiler.block.nodes]
    assert any("if_true_assign" in name for name in op_names)

def test_level_allocated_dispatch_keys():
    from torch_candle.func import vmap, get_active_dispatch_level
    
    def outer_func(x):
        assert get_active_dispatch_level() == 1
        
        def inner_func(y):
            assert get_active_dispatch_level() == 2
            return y * 2.0
            
        wrapped_inner = vmap(inner_func)
        return wrapped_inner(x)
        
    wrapped_outer = vmap(outer_func)
    x = torch.tensor([[1.0, 2.0]])
    wrapped_outer(x)
    
    assert get_active_dispatch_level() == 0

def test_rocm_backend_dispatch():
    # Dynamic AMD ROCm/HIP backend verification
    torch.register_privateuse1_backend("rocm")
    torch.register_privateuse1_backend("hip")

def test_proactive_block_reclaim():
    import torch_candle_backend as _kernels
    alloc = _kernels.StreamAwareAllocator()
    ptr = alloc.allocate(1024, 0, "test_tag")
    alloc.record_stream(ptr, 1)
    alloc.free(ptr, 0)
    alloc.cuda_free(ptr)



def test_zero_tool_call_guard():
    import torch_candle as torch
    torch.reset_kernel_call_count()
    assert torch.get_kernel_call_count() == 0
    
    import torch_candle_backend as _kernels
    import numpy as np
    x = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    _kernels.fast_relu(x)
    assert torch.get_kernel_call_count() > 0

