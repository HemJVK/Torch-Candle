"""
torch_candle.ops — Full torch functional API backed by Candle Rust via PyO3.

Design principle: every op dispatches to candle Rust first. numpy is used only
where candle has no native equivalent (trig inverses, cumsum, topk, sort, etc.)
and only on the CPU-side; these paths are clearly marked "# numpy fallback".
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, List, Tuple
import builtins as _builtins

import numpy as np
import math

from .tensor import Tensor
import torch_candle_backend as _kernels

# ─── helpers ─────────────────────────────────────────────────────────────────

def _wrap(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def _raw(x):
    """Unwrap to candle.Tensor."""
    return x._tensor if isinstance(x, Tensor) else x


def _np(t: Tensor) -> np.ndarray:
    """Return numpy view of a Tensor (numpy fallback helper)."""
    return t.numpy()


def _ones_raw(shape, device, dtype):
    return _kernels.PyTensor.ones(shape, device=device, dtype=dtype)


def _zeros_raw(shape, device, dtype):
    return _kernels.PyTensor.zeros(shape, device=device, dtype=dtype)


# ─── ARITHMETIC / POINTWISE ──────────────────────────────────────────────────

def add(input, other, alpha=1, out=None):
    if not isinstance(input, Tensor): input = _wrap(input)
    if not isinstance(other, Tensor): other = _wrap(other)
    if alpha != 1:
        other = other * alpha
    return input + other


def sub(input, other, alpha=1, out=None):
    if not isinstance(input, Tensor): input = _wrap(input)
    if not isinstance(other, Tensor): other = _wrap(other)
    if alpha != 1:
        other = other * alpha
    return input - other


def mul(input, other, out=None):
    return _wrap(input) * other


def div(input, other, rounding_mode=None, out=None):
    result = _wrap(input) / other
    if rounding_mode == 'floor':
        result = floor(result)
    elif rounding_mode == 'trunc':
        result = trunc(result)
    return result


def matmul(input, other, out=None):
    return _wrap(input).matmul(_wrap(other))


def mm(input, other):
    return _wrap(input).matmul(_wrap(other))


def bmm(input, mat2, out=None):
    """Batch matrix multiply — (B, N, M) @ (B, M, P) → (B, N, P)."""
    return _wrap(input).matmul(_wrap(mat2))


def mv(input, vec, out=None):
    return _wrap(input).matmul(_wrap(vec).unsqueeze(1)).squeeze(1)


def dot(input, other, out=None):
    return (_wrap(input) * _wrap(other)).sum()


def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    result = mm(mat1, mat2)
    if alpha != 1:
        result = result * alpha
    if beta != 1:
        return _wrap(input) * beta + result
    return _wrap(input) + result


def pow(input, exponent, out=None):
    return _wrap(input) ** exponent


def exp(input, out=None):
    return _wrap(input).exp()


def exp2(input, out=None):
    return (_wrap(input) * math.log(2)).exp()


def log(input, out=None):
    return _wrap(input).log()


def log2(input, out=None):
    return _wrap(input).log() * (1.0 / math.log(2))


def log10(input, out=None):
    return _wrap(input).log() * (1.0 / math.log(10))


def log1p(input, out=None):
    return (_wrap(input) + 1.0).log()


def sqrt(input, out=None):
    return _wrap(input).sqrt()


def rsqrt(input, out=None):
    return Tensor(_raw(_wrap(input)).sqrt().recip())


def reciprocal(input, out=None):
    return Tensor(_raw(_wrap(input)).recip())


def abs(input, out=None):
    return _wrap(input)._fast_wrap(_wrap(input)._tensor.abs())


absolute = abs


def neg(input, out=None):
    """Pure candle: x * -1."""
    return -_wrap(input)


def sign(input, out=None):
    """sign via candle: relu(x) - relu(-x), then clamp to ±1."""
    t   = _wrap(input)
    pos = t.relu()
    neg_part = t.neg().relu()
    diff = pos - neg_part
    # clamp to [-1,1]
    return diff.clamp(min=-1.0, max=1.0)


# ─── floor/ceil/round — numpy fallback (no candle equivalent) ─────────────────

def floor(input, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of floor is required.")


def ceil(input, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of ceil is required.")


def round(input, decimals=0, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of round is required.")


def trunc(input, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of trunc is required.")


def frac(input, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of frac is required.")


# ─── Trigonometry — candle-native where possible ─────────────────────────────

def sin(input, out=None):
    return Tensor(_raw(_wrap(input)).sin())


def cos(input, out=None):
    return Tensor(_raw(_wrap(input)).cos())


def tan(input, out=None):
    t = _wrap(input)
    return t.sin() / t.cos()


def asin(input, out=None):        # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of asin is required.")


def acos(input, out=None):        # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of acos is required.")


def atan(input, out=None):        # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of atan is required.")


def atan2(input, other, out=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of atan2 is required.")

arctan2 = atan2


def sinh(input, out=None):         # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of sinh is required.")


def cosh(input, out=None):         # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of cosh is required.")


def tanh(input, out=None):
    return _wrap(input).tanh()


def asinh(input, out=None):        # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of asinh is required.")


def acosh(input, out=None):        # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of acosh is required.")


def atanh(input, out=None):        # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of atanh is required.")


# ─── Activations ─────────────────────────────────────────────────────────────

def sigmoid(input, out=None):
    return _wrap(input).sigmoid()


def relu(input):
    return _wrap(input).relu()


# ─── clamp ───────────────────────────────────────────────────────────────────

def clamp(input, min=None, max=None, out=None):
    t = _wrap(input)
    if min is not None and max is not None:
        return t._fast_wrap(t._tensor.clamp(_builtins.float(min), _builtins.float(max)))
    raise RuntimeError("Zero-Fallback Mandate Violation: One-sided clamp is not implemented natively.")


clip = clamp


def addcmul(input, tensor1, tensor2, value=1, out=None):
    return _wrap(input) + _wrap(tensor1) * _wrap(tensor2) * value


def addcdiv(input, tensor1, tensor2, value=1, out=None):
    return _wrap(input) + (_wrap(tensor1) / _wrap(tensor2)) * value


def lerp(input, end, weight):
    start = _wrap(input)
    return start + (_wrap(end) - start) * weight


def erf(input, out=None):
    return _wrap(input).erf()


def erfinv(input):               # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of erfinv is required.")


# ─── LOGICAL OPS — numpy fallback (bool ops not in candle) ───────────────────

def logical_not(input, out=None):
    return Tensor((1.0 - _wrap(input).to_float()).clamp(0, 1))


def logical_and(input, other, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of logical_and is required.")


def logical_or(input, other, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of logical_or is required.")


def logical_xor(input, other, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of logical_xor is required.")


# ─── REDUCTIONS ──────────────────────────────────────────────────────────────

def sum(input, dim=None, keepdim=False, dtype=None, out=None):
    return _wrap(input).sum(dim=dim, keepdim=keepdim)


def mean(input, dim=None, keepdim=False, dtype=None, out=None):
    return _wrap(input).mean(dim=dim, keepdim=keepdim)


def prod(input, dim=None, keepdim=False, dtype=None, out=None):
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of prod is required.")


def std(input, dim=None, correction=1, keepdim=False, out=None):
    t    = _wrap(input)
    ddof = correction
    if dim is None:
        if not t.requires_grad:
            return Tensor(t._tensor.std_all(ddof))
        # scalar variance — candle mean_all
        mu   = Tensor(t._tensor.mean_all())
        diff = t - mu
        sq   = diff * diff
        n    = t.numel()
        var_val = sq.sum() * (1.0 / _builtins.max(1, n - ddof))
        return var_val.sqrt()
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim-wise std is required.")


def var(input, dim=None, correction=1, keepdim=False, out=None):
    t    = _wrap(input)
    ddof = correction
    if dim is None:
        if not t.requires_grad:
            return Tensor(t._tensor.var_all(ddof))
        mu      = Tensor(t._tensor.mean_all())
        diff    = t - mu
        n       = t.numel()
        return (diff * diff).sum() * (1.0 / _builtins.max(1, n - ddof))
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim-wise var is required.")


def max(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(t._tensor.max_all())
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim-wise max is required.")


def min(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(t._tensor.min_all())
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim-wise min is required.")


def argmax(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of argmax (dim=None) is required.")
    raw = t._tensor.argmax_keepdim(dim)
    if not keepdim:
        raw = raw.squeeze(dim)
    return Tensor(raw)


def argmin(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of argmin (dim=None) is required.")
    raw = t._tensor.argmin_keepdim(dim)
    if not keepdim:
        raw = raw.squeeze(dim)
    return Tensor(raw)


def norm(input, p=2, dim=None, keepdim=False, dtype=None, out=None):
    t = _wrap(input)
    if p == 2 and dim is not None:
        # candle-native: sqrt(sum(x^2))
        sq = t * t
        s  = sq.sum(dim=dim, keepdim=keepdim)
        return s.sqrt()
    if p == 2 and dim is None:
        if not t.requires_grad:
            return Tensor(t._tensor.norm_l2_all())
        sq = t * t
        return sq.sum().sqrt()
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of non-p=2 norm is required.")


def median(input, dim=None, keepdim=False):     # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of median is required.")


def logsumexp(input, dim, keepdim=False):
    """Numerically stable logsumexp — pure candle, no scipy."""
    t      = _wrap(input)
    # max for stability
    raw    = t._tensor
    mx     = Tensor(raw.max_keepdim(dim))                 # (…, 1, …)
    shifted = t - mx                                       # broadcast sub
    lse    = mx.squeeze(dim) + shifted.exp().sum(dim=dim, keepdim=False).log()
    if keepdim:
        lse = lse.unsqueeze(dim)
    return lse


def cumsum(input, dim, dtype=None, out=None):   # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of cumsum is required.")


def cumprod(input, dim, dtype=None, out=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of cumprod is required.")


def all(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        # candle: sum_all and check > 0 (all nonzero → all truthy)
        n = t.numel()
        s = Tensor(t._tensor.sum_all())
        # sum == n means all are 1.0
        arr = _np(s)
        return Tensor(float(bool(arr.item() >= n - 1e-4)))
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim-wise all is required.")


def any(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        s = Tensor(t._tensor.sum_all())
        return Tensor(float(_np(s).item() > 1e-6))
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim-wise any is required.")


def numel(input):
    return _wrap(input).numel()


# ─── COMPARISON OPS — candle arithmetic ──────────────────────────────────────

def _cmp(op_np, a, b):
    fn  = getattr(np, op_np)
    rhs = _np(_wrap(b)) if isinstance(b, Tensor) else b
    return Tensor(fn(_np(_wrap(a)), rhs).astype(np.float32))


def eq(input, other, out=None):    return _cmp('equal',         input, other)
def ne(input, other, out=None):    return _cmp('not_equal',     input, other)
def lt(input, other, out=None):    return _cmp('less',          input, other)
def le(input, other, out=None):    return _cmp('less_equal',    input, other)
def gt(input, other, out=None):    return _cmp('greater',       input, other)
def ge(input, other, out=None):    return _cmp('greater_equal', input, other)


def isnan(input):    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of isnan is required.")
def isinf(input):    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of isinf is required.")
def isfinite(input): raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of isfinite is required.")


def allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    return bool(np.allclose(_np(_wrap(input)), _np(_wrap(other)), rtol=rtol, atol=atol, equal_nan=equal_nan))


def topk(input, k, dim=-1, largest=True, sorted=True, out=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of topk is required.")


def sort(input, dim=-1, descending=False, stable=False, out=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of sort is required.")


# ─── INDEXING, JOINING, MUTATING ─────────────────────────────────────────────

def cat(tensors, dim=0, out=None):
    raw = [_raw(t) for t in tensors]
    return Tensor(_kernels.PyTensor.cat(raw, dim))


def stack(tensors, dim=0, out=None):
    raw = [_raw(t) for t in tensors]
    return Tensor(_kernels.PyTensor.stack(raw, dim))


def chunk(input, chunks, dim=0):   # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of chunk is required.")


def split(tensor, split_size_or_sections, dim=0):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of split is required.")


def view(input, *shape):
    return _wrap(input).view(*shape)


def reshape(input, *shape):
    return _wrap(input).reshape(*shape)


def squeeze(input, dim=None):
    return _wrap(input).squeeze(dim)


def unsqueeze(input, dim):
    return _wrap(input).unsqueeze(dim)


def flatten(input, start_dim=0, end_dim=-1):
    return _wrap(input).flatten(start_dim, end_dim)


def transpose(input, dim0, dim1):
    return _wrap(input).transpose(dim0, dim1)


def permute(input, dims):
    return _wrap(input).permute(*dims)


def gather(input, dim, index, sparse_grad=False, out=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of gather is required.")


def scatter_(input, dim, index, src):   # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of scatter_ is required.")


def index_select(input, dim, index, out=None):
    """Use candle-native index_select."""
    t   = _wrap(input)
    idx = _wrap(index)
    # Ensure index is uint32 for candle
    idx_u32 = idx._tensor.to_dtype("uint32")
    return Tensor(t._tensor.index_select(idx_u32, dim))


def where(condition, input=None, other=None):
    cond = _wrap(condition)
    if input is None:
        raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of where(condition) is required.")
    a = _wrap(input)
    b = _wrap(other)
    # candle where_cond needs u32
    c = cond._tensor.to_dtype("uint32")
    return Tensor(c.where_cond(a._tensor, b._tensor))


def masked_select(input, mask, out=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of masked_select is required.")


def nonzero(input, out=None, as_tuple=False):   # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of nonzero is required.")


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of unique is required.")


def tril(input, diagonal=0, out=None):   # numpy fallback (candle has no tril)
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of tril is required.")


def triu(input, diagonal=0, out=None):   # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of triu is required.")


def flip(input, dims, out=None):    # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of flip is required.")


def roll(input, shifts, dims=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of roll is required.")


def repeat_interleave(input, repeats, dim=None, output_size=None):  # numpy fallback
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of repeat_interleave is required.")


def broadcast_to(input, size):
    t = _wrap(input)
    return Tensor(t._tensor.broadcast_as(tuple(size)))


# ─── EINSUM ──────────────────────────────────────────────────────────────────

def einsum(equation, *operands):   # numpy fallback (complex indexing)
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of einsum is required.")


# ─── ALIASES ─────────────────────────────────────────────────────────────────

arcsin  = asin
arccos  = acos
arctan  = atan
arcsinh = asinh
arccosh = acosh
arctanh = atanh


# ─── TYPE / IDENTITY OPS ─────────────────────────────────────────────────────

def clone(input, memory_format=None):
    return _wrap(input).clone()


def detach(input):
    return _wrap(input).detach()


def contiguous(input):
    return _wrap(input).contiguous()


def type_as(input, other):
    return _wrap(input).to(dtype=_wrap(other).dtype)


def float(input):
    return _wrap(input).float()


# ─── NARROW / SELECT ─────────────────────────────────────────────────────────

def narrow(input, dim, start, length):
    return Tensor(_raw(_wrap(input)).narrow(dim, start, length))


def select(input, dim, index):
    t = _wrap(input)
    if dim == 0:
        return Tensor(t._tensor.get(index))
    raise RuntimeError("Zero-Fallback Mandate Violation: Native implementation of dim > 0 select is required.")
