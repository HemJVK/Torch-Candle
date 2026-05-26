import pytest
import pickle
import numpy as np
import torch_candle as torch
from torch_candle.utils import data

def test_multiprocessing_reduction():
    from torch_candle.multiprocessing import ForkingPickler
    
    # Test zero-copy multiprocessing pickling/unpickling
    x = torch.tensor([1.0, 2.0, 3.0])
    assert not x.is_shared()
    
    # Pickling automatically calls share_memory_() via ForkingPickler registration
    pickled = ForkingPickler.dumps(x)
    assert x.is_shared()
    
    # Unpickling reconstructs the exact same tensor values zero-copy
    y = ForkingPickler.loads(pickled)
    assert y.is_shared()
    np.testing.assert_array_equal(x.numpy(), y.numpy())
    assert y.requires_grad == x.requires_grad

def test_custom_samplers():
    # Test SubsetRandomSampler
    sampler = data.SubsetRandomSampler([0, 2, 4])
    assert len(sampler) == 3
    indices = list(sampler)
    assert set(indices) == {0, 2, 4}
    
    # Test WeightedRandomSampler
    w_sampler = data.WeightedRandomSampler([0.1, 0.9, 0.0], num_samples=10, replacement=True)
    assert len(w_sampler) == 10
    w_indices = list(w_sampler)
    assert len(w_indices) == 10
    assert 2 not in w_indices  # 0 weight index should never be selected

def test_functional_rearrange():
    from torch_candle.func import rearrange
    
    # Group decomposing rearrangement
    x = torch.arange(24).reshape(2, 3, 4)
    # 4 decomposed to 2x2
    y = rearrange(x, 'b c (h w) -> b h w c', h=2, w=2)
    assert tuple(y.shape) == (2, 2, 2, 3)
    
    # Flattening rearrangement
    z = rearrange(y, 'b h w c -> (b h w) c')
    assert tuple(z.shape) == (8, 3)
    np.testing.assert_array_equal(z.numpy()[0], [0, 4, 8])

def test_functional_autograd():
    from torch_candle.func import grad, vmap, vjp, jvp
    
    # Test grad helper
    def simple_fn(x):
        return x.sum()
        
    grad_fn = grad(simple_fn)
    x = torch.tensor([1.0, 2.0, 3.0])
    g = grad_fn(x)
    np.testing.assert_array_equal(g.numpy(), [1.0, 1.0, 1.0])
    
    # Test vmap
    vmap_fn = vmap(lambda a: a * 2.0)
    batch_in = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch_out = vmap_fn(batch_in)
    np.testing.assert_array_equal(batch_out.numpy(), [[2.0, 4.0], [6.0, 8.0]])

def _jit_test_fn(x, y):
    return x + y

def test_jit_tracing_and_saving(tmp_path):
    import torch_candle.jit as jit
    
    traced = jit.trace(_jit_test_fn)
    res = traced(torch.tensor(2.0), torch.tensor(3.0))
    assert res.item() == 5.0
    
    # Save and reload
    save_path = tmp_path / "model.pt"
    jit.save(traced, save_path)
    loaded = jit.load(save_path)
    assert loaded(torch.tensor(2.0), torch.tensor(3.0)).item() == 5.0

def test_c10_abstractions():
    import torch_candle.c10 as c10
    
    dev = c10.Device("cuda:1")
    assert dev.type == "cuda"
    assert dev.index == 1
    
    assert c10.DispatchKey.CPU == "CPU"
    assert c10.DispatchKey.MPS == "MPS"
    
    alloc = c10.get_allocator()
    assert alloc.name == "default"
    ptr = alloc.allocate(1024)
    assert "1024" in ptr

def test_aten_and_caffe2():
    import torch_candle.aten as aten
    import torch_candle.caffe2 as caffe2
    
    # Test aten ops
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    res = aten.add(x, y)
    np.testing.assert_array_equal(res.numpy(), [4.0, 6.0])
    
    assert not aten.mps.is_available()
    
    # Test caffe2
    net = caffe2.Net("EdgeNet")
    net.add_op("Add", ["x", "y"], ["z"])
    assert len(net.ops) == 1
    assert not caffe2.is_mobile_available()

def test_torchgen():
    from torch_candle.torchgen import generate_backend_signatures
    
    sig = generate_backend_signatures("add(Tensor self, Tensor other) -> Tensor")
    assert "#[pyo3(name = \"add\")]" in sig
    assert "self" in sig
    assert "other: &PyTensor" in sig

def test_distributed_collectives():
    import torch_candle.distributed as dist
    
    dist.init_process_group("gloo", rank=0, world_size=1)
    assert dist.is_initialized()
    assert dist.get_rank() == 0
    assert dist.get_world_size() == 1
    
    x = torch.tensor([1.0, 2.0, 3.0])
    # Single-process all_reduce is a no-op but executes cleanly
    res = dist.all_reduce(x)
    np.testing.assert_array_equal(res.numpy(), [1.0, 2.0, 3.0])
    
    dist.destroy_process_group()
    assert not dist.is_initialized()

def test_backends():
    import torch_candle.backends as backends
    
    assert backends.onednn.is_available()
    assert backends.onednn.enabled
    assert backends.mkl.is_available()
    assert not backends.mps.is_available()
    # CUDA is checked dynamically
    _ = backends.cuda.is_available()
    assert backends.cuda.matmul.allow_tf32
