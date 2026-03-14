"""
torch_candle.ops — Full torch functional API backed by Candle Rust via PyO3.

Design principle: every op dispatches to candle Rust first. numpy is used only
where candle has no native equivalent (trig inverses, cumsum, topk, sort, etc.)
and only on the CPU-side; these paths are clearly marked "# numpy fallback".
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, List, Tuple
import builtins as _builtins

import candle
import numpy as np
import math

from .tensor import Tensor

_f32 = candle.f32
_u8  = candle.u8

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
    return candle.ones(shape).to_device(device).to_dtype(dtype)


def _zeros_raw(shape, device, dtype):
    return candle.zeros(shape).to_device(device).to_dtype(dtype)


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
    """Pure candle: sqrt(sqr(x))."""
    return _wrap(input).abs()


absolute = abs


def neg(input, out=None):
    """Pure candle: x * -1."""
    return _wrap(input).neg()


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
    t = _wrap(input)
    return Tensor(np.floor(_np(t)).astype(np.float32))


def ceil(input, out=None):
    t = _wrap(input)
    return Tensor(np.ceil(_np(t)).astype(np.float32))


def round(input, decimals=0, out=None):
    t = _wrap(input)
    return Tensor(np.round(_np(t), decimals=decimals).astype(np.float32))


def trunc(input, out=None):
    t = _wrap(input)
    return Tensor(np.trunc(_np(t)).astype(np.float32))


def frac(input, out=None):
    t = _wrap(input)
    return t - trunc(t)


# ─── Trigonometry — candle-native where possible ─────────────────────────────

def sin(input, out=None):
    return Tensor(_raw(_wrap(input)).sin())


def cos(input, out=None):
    return Tensor(_raw(_wrap(input)).cos())


def tan(input, out=None):
    t = _wrap(input)
    return t.sin() / t.cos()


def asin(input, out=None):        # numpy fallback
    return Tensor(np.arcsin(_np(_wrap(input))).astype(np.float32))


def acos(input, out=None):        # numpy fallback
    return Tensor(np.arccos(_np(_wrap(input))).astype(np.float32))


def atan(input, out=None):        # numpy fallback
    return Tensor(np.arctan(_np(_wrap(input))).astype(np.float32))


def atan2(input, other, out=None):  # numpy fallback
    return Tensor(np.arctan2(_np(_wrap(input)), _np(_wrap(other))).astype(np.float32))

arctan2 = atan2


def sinh(input, out=None):         # numpy fallback
    return Tensor(np.sinh(_np(_wrap(input))).astype(np.float32))


def cosh(input, out=None):         # numpy fallback
    return Tensor(np.cosh(_np(_wrap(input))).astype(np.float32))


def tanh(input, out=None):
    return _wrap(input).tanh()


def asinh(input, out=None):        # numpy fallback
    return Tensor(np.arcsinh(_np(_wrap(input))).astype(np.float32))


def acosh(input, out=None):        # numpy fallback
    return Tensor(np.arccosh(_np(_wrap(input))).astype(np.float32))


def atanh(input, out=None):        # numpy fallback
    return Tensor(np.arctanh(_np(_wrap(input))).astype(np.float32))


# ─── Activations ─────────────────────────────────────────────────────────────

def sigmoid(input, out=None):
    return _wrap(input).sigmoid()


def relu(input):
    return _wrap(input).relu()


# ─── clamp ───────────────────────────────────────────────────────────────────

def clamp(input, min=None, max=None, out=None):
    return _wrap(input).clamp(min=min, max=max)


clip = clamp


def addcmul(input, tensor1, tensor2, value=1, out=None):
    return _wrap(input) + _wrap(tensor1) * _wrap(tensor2) * value


def addcdiv(input, tensor1, tensor2, value=1, out=None):
    return _wrap(input) + (_wrap(tensor1) / _wrap(tensor2)) * value


def lerp(input, end, weight):
    start = _wrap(input)
    return start + (_wrap(end) - start) * weight


def erf(input, out=None):
    # numpy fallback — no candle equivalent
    x   = _np(_wrap(input))
    approx = np.tanh((2.0 / np.sqrt(np.pi)) * (x + 0.111 * x**3))
    return Tensor(approx.astype(np.float32))


def erfinv(input):               # numpy fallback
    x  = _np(_wrap(input))
    a  = 8.0 * (np.pi - 3.0) / (3.0 * np.pi * (4.0 - np.pi))
    ln = np.log(1.0 - x**2 + 1e-7)
    t1 = 2.0 / (np.pi * a) + ln / 2.0
    return Tensor((np.sign(x) * np.sqrt(np.sqrt(t1**2 - ln / a) - t1)).astype(np.float32))


# ─── LOGICAL OPS — numpy fallback (bool ops not in candle) ───────────────────

def logical_not(input, out=None):
    return Tensor((1.0 - _wrap(input).to_float()).clamp(0, 1))


def logical_and(input, other, out=None):
    return Tensor(np.logical_and(_np(_wrap(input)), _np(_wrap(other))).astype(np.float32))


def logical_or(input, other, out=None):
    return Tensor(np.logical_or(_np(_wrap(input)), _np(_wrap(other))).astype(np.float32))


def logical_xor(input, other, out=None):
    return Tensor(np.logical_xor(_np(_wrap(input)), _np(_wrap(other))).astype(np.float32))


# ─── REDUCTIONS ──────────────────────────────────────────────────────────────

def sum(input, dim=None, keepdim=False, dtype=None, out=None):
    return _wrap(input).sum(dim=dim, keepdim=keepdim)


def mean(input, dim=None, keepdim=False, dtype=None, out=None):
    return _wrap(input).mean(dim=dim, keepdim=keepdim)


def prod(input, dim=None, keepdim=False, dtype=None, out=None):
    # numpy fallback (candle has no product reduction)
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.prod(_np(t))).real)
    return Tensor(np.prod(_np(t), axis=dim, keepdims=keepdim).astype(np.float32))


def std(input, dim=None, correction=1, keepdim=False, out=None):
    """Variance via candle: E[x^2] - E[x]^2, then sqrt."""
    t    = _wrap(input)
    ddof = correction
    if dim is None:
        # scalar variance — candle mean_all
        mu   = Tensor(t._tensor.mean_all())
        diff = t - mu
        sq   = diff * diff
        n    = t.numel()
        var_val = sq.sum() * (1.0 / _builtins.max(1, n - ddof))
        return var_val.sqrt()
    # numpy fallback for dim-wise (acceptable; not hot path)
    return Tensor(np.std(_np(t), axis=dim, ddof=ddof, keepdims=keepdim).astype(np.float32))


def var(input, dim=None, correction=1, keepdim=False, out=None):
    """Variance via candle arithmetic."""
    t    = _wrap(input)
    ddof = correction
    if dim is None:
        mu      = Tensor(t._tensor.mean_all())
        diff    = t - mu
        n       = t.numel()
        return (diff * diff).sum() * (1.0 / _builtins.max(1, n - ddof))
    return Tensor(np.var(_np(t), axis=dim, ddof=ddof, keepdims=keepdim).astype(np.float32))


def max(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        # cascade max_keepdim over all dims
        res = t._tensor
        for d in range(len(t.shape) - 1, -1, -1):
            res = res.max_keepdim(d)
        return Tensor(res)
    # numpy for (values, indices) tuple
    vals = Tensor(np.max(_np(t), axis=dim, keepdims=keepdim).astype(np.float32))
    idxs = Tensor(np.argmax(_np(t), axis=dim).astype(np.float32))
    return vals, idxs


def min(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        res = t._tensor
        for d in range(len(t.shape) - 1, -1, -1):
            res = res.min_keepdim(d)
        return Tensor(res)
    vals = Tensor(np.min(_np(t), axis=dim, keepdims=keepdim).astype(np.float32))
    idxs = Tensor(np.argmin(_np(t), axis=dim).astype(np.float32))
    return vals, idxs


def argmax(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.argmax(_np(t))))
    raw = t._tensor.argmax_keepdim(dim)
    if not keepdim:
        raw = raw.squeeze(dim)
    return Tensor(raw)


def argmin(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.argmin(_np(t))))
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
        sq = t * t
        return sq.sum().sqrt()
    # numpy fallback for other norms
    return Tensor(np.linalg.norm(_np(t), ord=p, axis=dim, keepdims=keepdim).astype(np.float32))


def median(input, dim=None, keepdim=False):     # numpy fallback
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.median(_np(t))))
    return Tensor(np.median(_np(t), axis=dim, keepdims=keepdim).astype(np.float32))


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
    return Tensor(np.cumsum(_np(_wrap(input)), axis=dim).astype(np.float32))


def cumprod(input, dim, dtype=None, out=None):  # numpy fallback
    return Tensor(np.cumprod(_np(_wrap(input)), axis=dim).astype(np.float32))


def all(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        # candle: sum_all and check > 0 (all nonzero → all truthy)
        n = t.numel()
        s = Tensor(t._tensor.sum_all())
        # sum == n means all are 1.0
        arr = _np(s)
        return Tensor(float(bool(arr.item() >= n - 1e-4)))
    return Tensor(np.all(_np(t), axis=dim, keepdims=keepdim).astype(np.float32))


def any(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        s = Tensor(t._tensor.sum_all())
        return Tensor(float(_np(s).item() > 1e-6))
    return Tensor(np.any(_np(t), axis=dim, keepdims=keepdim).astype(np.float32))


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


def isnan(input):    return Tensor(np.isnan(_np(_wrap(input))).astype(np.float32))
def isinf(input):    return Tensor(np.isinf(_np(_wrap(input))).astype(np.float32))
def isfinite(input): return Tensor(np.isfinite(_np(_wrap(input))).astype(np.float32))


def allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    return bool(np.allclose(_np(_wrap(input)), _np(_wrap(other)), rtol=rtol, atol=atol, equal_nan=equal_nan))


def topk(input, k, dim=-1, largest=True, sorted=True, out=None):  # numpy fallback
    t   = _wrap(input)
    arr = _np(t)
    if dim == -1:
        dim = arr.ndim - 1
    idx = np.argsort(arr, axis=dim)
    if largest:
        idx = np.flip(idx, axis=dim)
    idx  = np.take(idx, np.arange(k), axis=dim)
    vals = np.take_along_axis(arr, idx, axis=dim)
    return Tensor(vals.astype(np.float32)), Tensor(idx.astype(np.float32))


def sort(input, dim=-1, descending=False, stable=False, out=None):  # numpy fallback
    t   = _wrap(input)
    arr = _np(t)
    if dim == -1:
        dim = arr.ndim - 1
    idx  = np.argsort(arr, axis=dim, stable=stable)
    if descending:
        idx = np.flip(idx, axis=dim)
    vals = np.take_along_axis(arr, idx, axis=dim)
    return Tensor(vals.astype(np.float32)), Tensor(idx.astype(np.float32))


# ─── INDEXING, JOINING, MUTATING ─────────────────────────────────────────────

def cat(tensors, dim=0, out=None):
    raw = [t._tensor if isinstance(t, Tensor) else t for t in tensors]
    try:
        return Tensor(candle.cat(raw, dim))
    except Exception:
        arrs = [_np(_wrap(t)) for t in tensors]
        return Tensor(np.concatenate(arrs, axis=dim).astype(np.float32))


def stack(tensors, dim=0, out=None):
    raw = [t._tensor if isinstance(t, Tensor) else t for t in tensors]
    try:
        return Tensor(candle.stack(raw, dim))
    except Exception:
        arrs = [_np(_wrap(t)) for t in tensors]
        return Tensor(np.stack(arrs, axis=dim).astype(np.float32))


def chunk(input, chunks, dim=0):   # numpy fallback
    arrs = np.array_split(_np(_wrap(input)), chunks, axis=dim)
    return [Tensor(a.astype(np.float32)) for a in arrs]


def split(tensor, split_size_or_sections, dim=0):  # numpy fallback
    t   = _wrap(tensor)
    arr = _np(t)
    if isinstance(split_size_or_sections, int):
        indices = list(range(split_size_or_sections, arr.shape[dim], split_size_or_sections))
    else:
        indices = list(np.cumsum(split_size_or_sections)[:-1])
    return [Tensor(a.astype(np.float32)) for a in np.split(arr, indices, axis=dim)]


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
    return Tensor(np.take_along_axis(_np(_wrap(input)), _np(_wrap(index)).astype(np.int64), axis=dim).astype(np.float32))


def scatter_(input, dim, index, src):   # numpy fallback
    t   = _wrap(input)
    arr = _np(t).copy()
    idx = _np(_wrap(index)).astype(np.int64)
    src_arr = _np(_wrap(src)) if isinstance(src, Tensor) else src
    np.put_along_axis(arr, idx, src_arr, axis=dim)
    t._tensor = Tensor(arr.astype(np.float32))._tensor
    return t


def index_select(input, dim, index, out=None):
    """Use candle-native index_select."""
    t   = _wrap(input)
    idx = _wrap(index)
    return Tensor(t._tensor.index_select(_raw(idx).to_dtype(candle.u32), dim))


def where(condition, input=None, other=None):
    cond = _wrap(condition)
    if input is None:
        return Tensor(np.argwhere(_np(cond).astype(bool)).astype(np.float32))
    a_raw = _raw(_wrap(input))  if isinstance(input, Tensor) else input
    b_raw = _raw(_wrap(other))  if isinstance(other, Tensor) else other
    # Build u8 mask from cond (0./1. float → u8)
    cond_u8 = cond._tensor.to_dtype(_u8)
    # Ensure a_raw and b_raw are candle tensors
    if not isinstance(a_raw, candle.Tensor):
        a_raw = candle.ones(cond.shape).to_device(cond.device).to_dtype(cond.dtype) * float(a_raw)
    if not isinstance(b_raw, candle.Tensor):
        b_raw = candle.ones(cond.shape).to_device(cond.device).to_dtype(cond.dtype) * float(b_raw)
    return Tensor(cond_u8.where_cond(a_raw, b_raw))


def masked_select(input, mask, out=None):  # numpy fallback
    return Tensor(_np(_wrap(input))[_np(_wrap(mask)).astype(bool)].astype(np.float32))


def nonzero(input, out=None, as_tuple=False):   # numpy fallback
    indices = np.nonzero(_np(_wrap(input)))
    if as_tuple:
        return tuple(Tensor(i.astype(np.float32)) for i in indices)
    return Tensor(np.column_stack(indices).astype(np.float32))


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):  # numpy fallback
    result = np.unique(_np(_wrap(input)), return_inverse=return_inverse, return_counts=return_counts, axis=dim)
    if return_inverse or return_counts:
        return tuple(Tensor(r.astype(np.float32)) for r in result)
    return Tensor(result.astype(np.float32))


def tril(input, diagonal=0, out=None):   # numpy fallback (candle has no tril)
    return Tensor(np.tril(_np(_wrap(input)), k=diagonal).astype(np.float32))


def triu(input, diagonal=0, out=None):   # numpy fallback
    return Tensor(np.triu(_np(_wrap(input)), k=diagonal).astype(np.float32))


def flip(input, dims, out=None):    # numpy fallback
    return Tensor(np.flip(_np(_wrap(input)), axis=dims).copy().astype(np.float32))


def roll(input, shifts, dims=None):  # numpy fallback
    return Tensor(np.roll(_np(_wrap(input)), shifts, axis=dims).astype(np.float32))


def repeat_interleave(input, repeats, dim=None, output_size=None):  # numpy fallback
    return Tensor(np.repeat(_np(_wrap(input)), repeats, axis=dim).astype(np.float32))


def broadcast_to(input, size):
    t = _wrap(input)
    try:
        return Tensor(t._tensor.broadcast_as(tuple(size)))
    except Exception:
        return Tensor(np.broadcast_to(_np(t), size).copy().astype(np.float32))


# ─── EINSUM ──────────────────────────────────────────────────────────────────

def einsum(equation, *operands):   # numpy fallback (complex indexing)
    ops_np = [_np(_wrap(o)) for o in operands]
    return Tensor(np.einsum(equation, *ops_np).astype(np.float32))


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
    t   = _wrap(input)
    out = Tensor(t._tensor)
    out.requires_grad = False
    out.grad_fn       = None
    return out


def contiguous(input):
    return Tensor(_raw(_wrap(input)).contiguous())


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
    return Tensor(np.take(_np(t), index, axis=dim).astype(np.float32))
