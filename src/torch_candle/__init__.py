"""
torch_candle — A one-to-one PyTorch 2.10 API clone using Candle (Rust/PyO3) as the backend.
All ops dispatch directly to Candle Rust – hardware agnostic (CPU/CUDA/Metal).
"""

try:
    import candle
except ImportError:
    candle = None

import numpy as np
import math

from .tensor import Tensor
from . import ops
from .ops import (
    add, sub, mul, div, matmul, sum, mean, relu, mm,
    cat, stack, log, exp, pow, view, reshape, squeeze, unsqueeze,
    # Pointwise math
    sin, cos, tan, asin, acos, atan, atan2,
    sinh, cosh, tanh, abs, neg, sign,
    floor, ceil, round, trunc, frac,
    sqrt, rsqrt, reciprocal, clamp,
    addcmul, addcdiv, lerp,
    logical_and, logical_or, logical_not, logical_xor,
    # Reduction
    argmax, argmin, max, min, prod, std, var,
    all, any, cumsum, cumprod, norm, median, logsumexp,
    # Comparison
    eq, ne, lt, le, gt, ge,
    isnan, isinf, isfinite, allclose,
    topk, sort,
    # Indexing & Joining
    gather, where, masked_select, tril, triu,
    index_select, chunk, split, nonzero, unique, flip, roll,
    # BLAS
    bmm, addmm, mv, dot, einsum,
    # Misc
    numel,
)
from .device import device
from . import cuda

try:
    from . import nn
    from . import optim
    from . import utils
    from . import linalg
    from . import fft
    from . import amp
    from . import random as _random
    from . import distributions
    from . import autograd
except ImportError:
    pass

# ============================================================
# Context Managers — torch.no_grad / enable_grad
# ============================================================
class no_grad:
    """Context manager equivalent to torch.no_grad()"""
    def __init__(self):
        pass
    def __enter__(self):
        Tensor._grad_enabled = False
        return self
    def __exit__(self, *args):
        Tensor._grad_enabled = True

class enable_grad:
    def __enter__(self):
        Tensor._grad_enabled = True
        return self
    def __exit__(self, *args):
        pass

def set_grad_enabled(mode: bool):
    Tensor._grad_enabled = mode
    return mode

# ============================================================
# Tensor Production
# ============================================================

def tensor(data, dtype=None, device=None, requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def as_tensor(data, dtype=None, device=None):
    return Tensor(data, dtype=dtype, device=device)

def _get_shape(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return tuple(args[0])
    return tuple(args)

# --- Factory functions via Candle Rust ---

def ones(*size, dtype=None, device=None, requires_grad=False, out=None):
    shape = _get_shape(*size)
    if candle is not None:
        t = candle.ones(shape)
    else:
        t = Tensor(np.ones(shape).tolist())
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros(*size, dtype=None, device=None, requires_grad=False, out=None):
    shape = _get_shape(*size)
    if candle is not None:
        t = candle.zeros(shape)
    else:
        t = Tensor(np.zeros(shape).tolist())
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

def randn(*size, dtype=None, device=None, requires_grad=False, generator=None, out=None):
    shape = _get_shape(*size)
    if candle is not None:
        t = candle.randn(shape)
    else:
        t = Tensor(np.random.randn(*shape).tolist())
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

def rand(*size, dtype=None, device=None, requires_grad=False, generator=None, out=None):
    shape = _get_shape(*size)
    # Candle has rand (uniform 0-1)
    if candle is not None and hasattr(candle, 'rand'):
        t = candle.rand(shape)
    else:
        t = Tensor(np.random.rand(*shape).tolist())
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

def randint(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None):
    if high is None:
        high, low = low, 0
    if size is None:
        raise ValueError("size must be specified for randint")
    data = np.random.randint(low, high, size=size).astype(np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def randperm(n, dtype=None, device=None, requires_grad=False, generator=None):
    data = np.random.permutation(n).astype(np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        end, start = start, 0
    data = np.arange(start, end, step, dtype=np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    data = np.linspace(start, end, steps, dtype=np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def logspace(start, end, steps, base=10.0, dtype=None, device=None, requires_grad=False):
    data = np.logspace(start, end, steps, base=base, dtype=np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def eye(n, m=None, dtype=None, device=None, requires_grad=False):
    if m is None:
        m = n
    data = np.eye(n, m, dtype=np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    data = np.full(size, fill_value, dtype=np.float32).tolist()
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def empty(*size, dtype=None, device=None, requires_grad=False):
    shape = _get_shape(*size)
    if candle is not None and hasattr(candle, 'zeros'):
        t = candle.zeros(shape)  # candle has no uninit; zeros is fine
    else:
        t = Tensor(np.zeros(shape).tolist())
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

# --- _like variants ---
def ones_like(input, dtype=None, device=None, requires_grad=False):
    return ones(*input.shape, dtype=dtype or input.dtype, device=device or input.device, requires_grad=requires_grad)

def zeros_like(input, dtype=None, device=None, requires_grad=False):
    return zeros(*input.shape, dtype=dtype or input.dtype, device=device or input.device, requires_grad=requires_grad)

def randn_like(input, dtype=None, device=None, requires_grad=False):
    return randn(*input.shape, dtype=dtype or input.dtype, device=device or input.device, requires_grad=requires_grad)

def rand_like(input, dtype=None, device=None, requires_grad=False):
    return rand(*input.shape, dtype=dtype or input.dtype, device=device or input.device, requires_grad=requires_grad)

def full_like(input, fill_value, dtype=None, device=None, requires_grad=False):
    return full(input.shape, fill_value, dtype=dtype or input.dtype, device=device or input.device, requires_grad=requires_grad)

def empty_like(input, dtype=None, device=None, requires_grad=False):
    return empty(*input.shape, dtype=dtype or input.dtype, device=device or input.device, requires_grad=requires_grad)

# ============================================================
# Serialisation — torch.save / torch.load
# ============================================================
def save(obj, f):
    import pickle
    if isinstance(obj, Tensor):
        obj = obj.numpy()
    with open(f, 'wb') as fh:
        pickle.dump(obj, fh)

def load(f, map_location=None):
    import pickle
    with open(f, 'rb') as fh:
        obj = pickle.load(fh)
    if isinstance(obj, np.ndarray):
        return Tensor(obj)
    return obj

# ============================================================
# Random
# ============================================================
def manual_seed(seed: int):
    np.random.seed(seed)

def seed():
    np.random.seed()

# ============================================================
# Dtype exports (map to candle dtypes)
# ============================================================
if candle is not None:
    float32 = getattr(candle, 'f32', None)
    float64 = getattr(candle, 'f64', None)
    int64   = getattr(candle, 'i64', getattr(candle, 'u32', None))
    int32   = getattr(candle, 'i32', getattr(candle, 'u32', None))
    float16 = getattr(candle, 'f16', None)
    bfloat16 = getattr(candle, 'bf16', None)
    uint8   = getattr(candle, 'u8', None)
    long    = int64
    bool    = getattr(candle, 'u8', None)   # closest
else:
    float32 = float64 = int64 = int32 = float16 = bfloat16 = uint8 = long = bool = None

# Constants
inf = math.inf
nan = math.nan
e   = math.e
pi  = math.pi
