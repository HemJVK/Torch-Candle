import pytest
import os
import multiprocessing
import numpy as np
import torch_candle as torch
from torch_candle.multiprocessing import ForkingPickler

# 1. Shared Memory IPC Stability
def _ipc_worker_read(conn):
    try:
        t = conn.recv()
        # Verify tensor integrity on the receiving side
        assert t.is_shared(), "Tensor in worker is not marked as shared"
        assert t.shape == (3,), f"Unexpected shape: {t.shape}"
        assert np.allclose(t.numpy(), [1.0, 2.0, 3.0]), f"Unexpected values: {t.numpy()}"
        conn.send((True, t.numpy().tolist()))
    except Exception as e:
        conn.send((False, str(e)))

def test_shared_memory_ipc_stability():
    parent_conn, child_conn = multiprocessing.Pipe()
    
    t = torch.tensor([1.0, 2.0, 3.0])
    t.share_memory_()
    assert t.is_shared(), "Tensor is not marked as shared on parent side"
    
    p = multiprocessing.Process(target=_ipc_worker_read, args=(child_conn,))
    p.start()
    
    parent_conn.send(t)
    
    success, result = parent_conn.recv()
    p.join()
    
    assert success, f"Worker process verification failed: {result}"
    assert result == [1.0, 2.0, 3.0], f"Recovered values mismatch: {result}"

def test_sha_numerical_stability():
    # Test standard gradient propagation of NaNs (SHA is disabled)
    torch.Tensor.enable_sha = False
    torch.set_disable_ema_estimates(False)
    torch.clear_grad_history()

    w_std = torch.Tensor([5.0], requires_grad=True)
    loss1 = w_std * 2.0
    loss1.backward()
    
    # Inject NaN directly simulating mathematical explosion
    w_std.grad = torch.Tensor([float('nan')])
    assert np.isnan(w_std.grad.item()), "NaN was not retained"

    # Test standard gradient propagation of NaNs (SHA is enabled, but EMA is decommissioned)
    torch.Tensor.enable_sha = True
    torch.set_disable_ema_estimates(False)
    torch.clear_grad_history()

    w_sha = torch.Tensor([5.0], requires_grad=True)
    loss_healthy = w_sha * 3.0
    loss_healthy.backward()
    assert w_sha.grad.item() == pytest.approx(3.0)
    
    # Inject anomaly gradient (NaN)
    w_sha.grad = torch.Tensor([float('nan')])
    assert np.isnan(w_sha.grad.item()), "NaN was not propagated naturally under decommissioned EMA"

# 3. Auto-Device Alignment Discovery & Stress Test
def test_auto_device_alignment_stress():
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create operand 1 on target device
    x = torch.tensor([10.0, 20.0], device=target_device)
    
    # Create operand 2 on standard CPU device
    y = torch.tensor([2.0, 3.0], device="cpu")
    
    # Addition with mixed devices: this would crash in standard PyTorch, but Torch-Candle aligns them!
    res_add = x + y
    assert res_add.device == target_device, "Output tensor device mismatch"
    np.testing.assert_allclose(res_add.numpy(), [12.0, 23.0])
    
    # Subtraction mixed alignment
    res_sub = x - y
    assert res_sub.device == target_device
    np.testing.assert_allclose(res_sub.numpy(), [8.0, 17.0])
    
    # Multiplication mixed alignment
    res_mul = x * y
    assert res_mul.device == target_device
    np.testing.assert_allclose(res_mul.numpy(), [20.0, 60.0])

def simple_model(a, b):
    return a * b + 5.0

# 4. Graph Export & Load Integrity (JIT compilation vs eager mode)
def test_graph_export_integrity(tmp_path):
    import torch_candle.jit as jit
    
    traced = jit.trace(simple_model)
    
    x = torch.tensor([2.0])
    y = torch.tensor([4.0])
    
    eager_out = simple_model(x, y)
    traced_out = traced(x, y)
    
    assert traced_out.item() == eager_out.item(), "Traced JIT output does not match eager output"
    
    # Serialize to computational graph file
    filepath = tmp_path / "traced_graph.pt"
    jit.save(traced, filepath)
    assert os.path.exists(filepath), "Graph serialization file was not created"
    
    # Load the serialized graph back
    loaded = jit.load(filepath)
    
    # Execute the loaded graph
    loaded_out = loaded(x, y)
    assert loaded_out.item() == eager_out.item(), "Loaded serialized graph output does not match eager output"

