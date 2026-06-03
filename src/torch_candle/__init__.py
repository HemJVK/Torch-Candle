import numpy as np
import math
import os
os.environ["MALLOC_MMAP_THRESHOLD_"] = "65536"

import torch_candle_backend as _kernels

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

try: from . import nn
except ImportError: pass

try: from . import optim
except ImportError: pass

try: from . import utils
except ImportError: pass

try: from . import linalg
except ImportError: pass

try: from . import fft
except ImportError: pass

try: from . import amp
except ImportError: pass

try: from . import random as _random
except ImportError: pass

try: from . import distributions
except ImportError: pass

try: from . import autograd
except ImportError: pass

try: from . import multiprocessing
except ImportError: pass

try:
    from . import func
    from .func import (
        subclass_dispatch, AttnBiasTensor, jacrev, jacfwd, hessian,
        DynamicSubclassDispatcher, make_functional, make_functional_with_buffers, vmap, grad
    )
except ImportError:
    pass

try:
    from .ast_parser import RustASTParser
    from torch_candle_backend import NativeASTParser, VmapDispatcher
except ImportError:
    pass

try: from . import jit
except ImportError: pass

try: from . import c10
except ImportError: pass

try: from . import aten
except ImportError: pass

try: from . import caffe2
except ImportError: pass

try: from . import torchgen
except ImportError: pass

try: from . import distributed
except ImportError: pass

try: from . import backends
except ImportError: pass

# ============================================================
# Decentralized Backend Dispatch Registry APIs
# ============================================================
def register_privateuse1_backend(backend_name: str):
    """
    Statically or dynamically register a new hardware backend for Torch-Candle.
    Exposes stable public APIs matching torch.register_privateuse1_backend.
    """
    _kernels.PyDispatchRegistry.register_backend(backend_name)

def register_kernel(op_name: str, backend_name: str, kernel):
    """
    Register a dynamic dispatch kernel for a specific operator and hardware backend.
    """
    _kernels.PyDispatchRegistry.register_kernel(op_name, backend_name, kernel)

def dispatch_kernel(op_name: str, backend_name: str, *args):
    """
    Dynamically dispatch an operator to a registered backend kernel.
    """
    return _kernels.PyDispatchRegistry.dispatch(op_name, backend_name, args)

def clear_grad_history():
    """
    Clear the running gradient histories used by Self-Healing Autograd.
    """
    _kernels.clear_grad_history()
    Tensor._grad_history.clear()

class HardValidationFailure(Exception):
    """
    Raised when an agent attempts to game autograd validation using bypass mechanisms
    or failing actual gradient reconstruction calls during step phases.
    """
    pass

class ZeroToolCallGuard:
    """
    Middleware guard to verify that the agent execution actually invoked tools in the sandbox
    rather than reporting a phantom success.
    """
    _tool_call_count = 0

    @classmethod
    def increment_tool_call_count(cls):
        cls._tool_call_count += 1

    @classmethod
    def reset_tool_call_count(cls):
        cls._tool_call_count = 0

    @classmethod
    def get_tool_call_count(cls) -> int:
        return cls._tool_call_count

    @classmethod
    def verify_execution(cls, agent_state: str):
        """
        Validates terminal state. If 'Success' but tool calls are 0, triggers HardValidationFailure nudge.
        """
        if agent_state.lower() == "success" and cls._tool_call_count == 0:
            raise HardValidationFailure(
                "🚨 [Zero-Tool-Call Guard] Phantom Agent detected: "
                "Agent reported 'Success' but executed 0 tool calls! Triggering automatic nudge/retry."
            )

DISABLE_EMA_ESTIMATES = False

def set_disable_ema_estimates(val: bool):
    global DISABLE_EMA_ESTIMATES
    DISABLE_EMA_ESTIMATES = val

def get_disable_ema_estimates() -> bool:
    global DISABLE_EMA_ESTIMATES
    return DISABLE_EMA_ESTIMATES

def get_kernel_call_count() -> int:
    return _kernels.get_kernel_call_count()

def reset_kernel_call_count():
    _kernels.reset_kernel_call_count()

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

class standard_mode:
    """Context manager to run standard autograd without self-healing reconstruction."""
    def __init__(self):
        self.prev = True
    def __enter__(self):
        self.prev = getattr(Tensor, "enable_sha", True)
        Tensor.enable_sha = False
        return self
    def __exit__(self, *args):
        Tensor.enable_sha = self.prev

# ============================================================
# Tensor Production
# ============================================================

def tensor(data, dtype=None, device=None, requires_grad=False):
    """Create a tensor from data."""
    if device is None: device = "cpu"
    if dtype is None: dtype = "float32"
    if isinstance(data, Tensor):
        return data
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def as_tensor(data, dtype=None, device=None):
    if device is None: device = "cpu"
    if dtype is None: dtype = "float32"
    return Tensor(data, dtype=dtype, device=device)

def _get_shape(*args):
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return tuple(args[0])
    return tuple(args)

# --- Factory functions via Candle Rust ---

def ones(*size, dtype="float32", device="cpu", requires_grad=False, out=None):
    """All-ones tensor."""
    shape = _get_shape(*size)
    return Tensor(_kernels.PyTensor.ones(shape, device=device, dtype=dtype), requires_grad=requires_grad)

def zeros(*size, dtype="float32", device="cpu", requires_grad=False, out=None):
    """All-zeros tensor."""
    if device is None: device = "cpu"
    if dtype is None: dtype = "float32"
    shape = _get_shape(*size)
    return Tensor(_kernels.PyTensor.zeros(shape, device=device, dtype=dtype), requires_grad=requires_grad)

def randn(*size, dtype="float32", device="cpu", requires_grad=False, generator=None, out=None):
    """Standard-normal tensor via native Rust."""
    shape = _get_shape(*size)
    return Tensor(_kernels.PyTensor.randn(shape, device=device, dtype=dtype), requires_grad=requires_grad)

def rand(*size, dtype="float32", device="cpu", requires_grad=False, generator=None, out=None):
    """Uniform [0,1) tensor via native Rust."""
    shape = _get_shape(*size)
    return Tensor(_kernels.PyTensor.rand(shape, device=device, dtype=dtype), requires_grad=requires_grad)

def randint(low, high=None, size=None, dtype=None, device=None, requires_grad=False, generator=None):
    if high is None:
        high, low = low, 0
    if size is None:
        raise ValueError("size must be specified for randint")
    arr = np.random.randint(low, high, size=size).astype(np.float32)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def randperm(n, dtype=None, device=None, requires_grad=False, generator=None):
    arr = np.random.permutation(n).astype(np.float32)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def arange(start, end=None, step=1, dtype=None, device=None, requires_grad=False):
    if end is None:
        end, start = start, 0
    arr = np.arange(start, end, step, dtype=np.float32)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def linspace(start, end, steps, dtype=None, device=None, requires_grad=False):
    arr = np.linspace(start, end, steps, dtype=np.float32)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def logspace(start, end, steps, base=10.0, dtype=None, device=None, requires_grad=False):
    arr = np.logspace(start, end, steps, base=base, dtype=np.float32)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def eye(n, m=None, dtype=None, device=None, requires_grad=False):
    """Identity matrix — candle zeros + diagonal fill via ops."""
    if m is None:
        m = n
    arr = np.eye(n, m, dtype=np.float32)
    return Tensor(arr, dtype=dtype, device=device, requires_grad=requires_grad)

def full(size, fill_value, dtype=None, device=None, requires_grad=False):
    """Constant-filled tensor."""
    return ones(*size, dtype=dtype, device=device, requires_grad=requires_grad) * float(fill_value)

def empty(*size, dtype=None, device=None, requires_grad=False):
    """Uninitialised (zero-initialised) tensor."""
    return zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)

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
    with open(f, 'wb') as fh:
        pickle.dump(obj, fh)

def load(f, map_location=None):
    import pickle
    with open(f, 'rb') as fh:
        return pickle.load(fh)

# ============================================================
# Dynamic Graph Compilation JIT & PyTorch Compatibility Layer
# ============================================================
from .compile import compile
from .compat import enable_torch_compat

# ============================================================
# Random
# ============================================================
def manual_seed(seed: int):
    np.random.seed(seed)

def seed():
    np.random.seed()

# ============================================================
# Dtype exports (map to candle dtypes)
# ============================================================# Dtype exports (string mappings)
float32 = "float32"
float64 = "float64"
int64   = "int64"
int32   = "int32"
float16 = "float16"
bfloat16 = "bfloat16"
uint8   = "uint8"
long    = "int64"
bool    = "bool"

# Constants
inf = math.inf
nan = math.nan
e   = math.e
pi  = math.pi
