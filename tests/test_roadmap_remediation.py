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
    
    # 5. Test Tensor level record_stream
    t = torch.tensor([1.0, 2.0, 3.0])
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
    
    # Third pass: dynamic shape (trigger eager mode fallback)
    c = torch.tensor([1.0, 2.0])
    d = torch.tensor([3.0, 4.0])
    out3 = compiled(c, d)
    assert np.allclose(out3.numpy(), [4.0, 6.0])
    
    captured = capsys.readouterr()
    assert "Dynamic shape detected" in captured.out


# ============================================================
# Phase 5: Concurrency & SPSC Ring Buffer Tests
# ============================================================
def test_spsc_ring_buffer():
    buf = _kernels.SPSCRingBuffer()
    results = []
    
    def consumer():
        try:
            task = buf.wait_and_pop()
            results.append((True, task.op_code, task.device_id, bytes(task.payload).decode().rstrip('\x00')))
        except Exception as e:
            results.append((False, str(e)))
            
    t = threading.Thread(target=consumer)
    t.start()
    
    # Wait a bit, then push item to unpark/trigger consumer
    time.sleep(0.1)
    buf.push(42, 1, b"Hello SPSC IPC")
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
