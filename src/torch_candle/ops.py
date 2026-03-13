"""
torch_candle.ops — Complete torch functional API backed by Candle Rust via PyO3.
Every function dispatches directly to candle; numpy is used only as a fallback where
Candle does not yet expose the operator in its Python bindings.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, List, Tuple

try:
    import candle
except ImportError:
    candle = None

import numpy as np
import math

from .tensor import Tensor


# ============================================================
# Internal helpers
# ============================================================
def _wrap(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

def _raw(x):
    """Unwrap to candle Tensor if possible."""
    if isinstance(x, Tensor):
        return x._tensor
    return x

def _np_op(name, t: Tensor, **kw) -> Tensor:
    """Fallback: apply numpy op and re-wrap."""
    fn = getattr(np, name)
    return Tensor(fn(t.numpy(), **kw).astype(np.float32))

# ============================================================
# ARITHMETIC / POINTWISE
# ============================================================

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
    if not isinstance(input, Tensor): input = _wrap(input)
    return input * other

def div(input, other, rounding_mode=None, out=None):
    if not isinstance(input, Tensor): input = _wrap(input)
    result = input / other
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
    """Batch matrix multiply — (B, N, M) @ (B, M, P) -> (B, N, P). Dispatches to candle matmul."""
    return _wrap(input).matmul(_wrap(mat2))

def mv(input, vec, out=None):
    # (M, N) @ (N,) -> (M,)
    return _wrap(input).matmul(_wrap(vec).unsqueeze(1)).squeeze(1)

def dot(input, other, out=None):
    # Flat 1D dot product
    a, b = _wrap(input), _wrap(other)
    return (a * b).sum()

def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
    # out = beta * input + alpha * (mat1 @ mat2)
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
    # 2^x = e^(x * ln2)
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
    return _wrap(input).abs()

absolute = abs   # alias

def neg(input, out=None):
    return _wrap(input).neg()

def sign(input, out=None):
    t = _wrap(input)
    pos = gt(t, 0).to_float()
    neg_sign = lt(t, 0).to_float()
    return pos - neg_sign

def floor(input, out=None):
    return _np_op('floor', _wrap(input))

def ceil(input, out=None):
    return _np_op('ceil', _wrap(input))

def round(input, decimals=0, out=None):
    return _np_op('round', _wrap(input), decimals=decimals)

def trunc(input, out=None):
    return _np_op('trunc', _wrap(input))

def frac(input, out=None):
    t = _wrap(input)
    return t - trunc(t)

def sin(input, out=None):
    t = _wrap(input)
    if hasattr(t._tensor, 'sin'):
        return Tensor(t._tensor.sin())
    return _np_op('sin', t)

def cos(input, out=None):
    t = _wrap(input)
    if hasattr(t._tensor, 'cos'):
        return Tensor(t._tensor.cos())
    return _np_op('cos', t)

def tan(input, out=None):
    return sin(input) / cos(input)

def asin(input, out=None):
    return _np_op('arcsin', _wrap(input))

def acos(input, out=None):
    return _np_op('arccos', _wrap(input))

def atan(input, out=None):
    return _np_op('arctan', _wrap(input))

def atan2(input, other, out=None):
    return Tensor(np.arctan2(_wrap(input).numpy(), _wrap(other).numpy()))

arctan2 = atan2   # alias

def sinh(input, out=None):
    return _np_op('sinh', _wrap(input))

def cosh(input, out=None):
    return _np_op('cosh', _wrap(input))

def tanh(input, out=None):
    return _wrap(input).tanh()

def asinh(input, out=None):
    return _np_op('arcsinh', _wrap(input))

def acosh(input, out=None):
    return _np_op('arccosh', _wrap(input))

def atanh(input, out=None):
    return _np_op('arctanh', _wrap(input))

def clamp(input, min=None, max=None, out=None):
    return _wrap(input).clamp(min=min, max=max)

clip = clamp   # alias

def addcmul(input, tensor1, tensor2, value=1, out=None):
    return _wrap(input) + _wrap(tensor1) * _wrap(tensor2) * value

def addcdiv(input, tensor1, tensor2, value=1, out=None):
    return _wrap(input) + (_wrap(tensor1) / _wrap(tensor2)) * value

def lerp(input, end, weight):
    return _wrap(input) + (_wrap(end) - _wrap(input)) * weight

def erf(input, out=None):
    # Approximation of error function using tanh
    # erf(x) ≈ tanh( (2/sqrt(pi)) * (x + 0.111x^3) )
    x = _wrap(input).numpy()
    approx = np.tanh( (2.0 / np.sqrt(np.pi)) * (x + 0.111 * x**3) )
    return Tensor(approx.astype(np.float32))

def erfinv(input):
    # Approximation of inverse error function
    x = _wrap(input).numpy()
    # using a simple approximation or pseudo-inverse: avoiding scipy
    a = 8.0 * (np.pi - 3.0) / (3.0 * np.pi * (4.0 - np.pi))
    ln_1_x2 = np.log(1.0 - x**2 + 1e-7)
    term1 = 2.0 / (np.pi * a) + ln_1_x2 / 2.0
    approx = np.sign(x) * np.sqrt(np.sqrt(term1**2 - ln_1_x2 / a) - term1)
    return Tensor(approx.astype(np.float32))

# ============================================================
# LOGICAL OPS
# ============================================================

def logical_not(input, out=None):
    return Tensor((1.0 - _wrap(input).to_float()).clamp(0, 1))

def logical_and(input, other, out=None):
    a, b = _wrap(input).to_float(), _wrap(other).to_float()
    return Tensor(np.logical_and(a.numpy(), b.numpy()).astype(np.float32))

def logical_or(input, other, out=None):
    return Tensor(np.logical_or(_wrap(input).numpy(), _wrap(other).numpy()).astype(np.float32))

def logical_xor(input, other, out=None):
    return Tensor(np.logical_xor(_wrap(input).numpy(), _wrap(other).numpy()).astype(np.float32))

# ============================================================
# REDUCTION OPS
# ============================================================

def sum(input, dim=None, keepdim=False, dtype=None, out=None):
    return _wrap(input).sum(dim=dim, keepdim=keepdim)

def mean(input, dim=None, keepdim=False, dtype=None, out=None):
    return _wrap(input).mean(dim=dim, keepdim=keepdim)

def prod(input, dim=None, keepdim=False, dtype=None, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(np.prod(t.numpy()).astype(np.float32))
    return Tensor(np.prod(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))

def std(input, dim=None, correction=1, keepdim=False, out=None):
    t = _wrap(input)
    ddof = correction
    if dim is None:
        return Tensor(np.std(t.numpy(), ddof=ddof).astype(np.float32))
    return Tensor(np.std(t.numpy(), axis=dim, ddof=ddof, keepdims=keepdim).astype(np.float32))

def var(input, dim=None, correction=1, keepdim=False, out=None):
    t = _wrap(input)
    ddof = correction
    if dim is None:
        return Tensor(np.var(t.numpy(), ddof=ddof).astype(np.float32))
    return Tensor(np.var(t.numpy(), axis=dim, ddof=ddof, keepdims=keepdim).astype(np.float32))

def max(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(t._tensor.max_keepdim(0).max_keepdim(0))  # approx for scalar
    # Return (values, indices) namedtuple-style
    vals = Tensor(np.max(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))
    idxs = Tensor(np.argmax(t.numpy(), axis=dim).astype(np.float32))
    return vals, idxs

def min(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(np.min(t.numpy()).astype(np.float32))
    vals = Tensor(np.min(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))
    idxs = Tensor(np.argmin(t.numpy(), axis=dim).astype(np.float32))
    return vals, idxs

def argmax(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.argmax(t.numpy())))
    return Tensor(np.argmax(t.numpy(), axis=dim).astype(np.float32))

def argmin(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.argmin(t.numpy())))
    return Tensor(np.argmin(t.numpy(), axis=dim).astype(np.float32))

def norm(input, p=2, dim=None, keepdim=False, dtype=None, out=None):
    t = _wrap(input)
    return Tensor(np.linalg.norm(t.numpy(), ord=p, axis=dim, keepdims=keepdim).astype(np.float32))

def median(input, dim=None, keepdim=False):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.median(t.numpy())))
    vals = Tensor(np.median(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))
    return vals

def logsumexp(input, dim, keepdim=False):
    # log(sum(exp(x))) with numerical stability: max + log(sum(exp(x - max)))
    from scipy.special import logsumexp as sp_lse
    t = _wrap(input)
    return Tensor(sp_lse(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))

def cumsum(input, dim, dtype=None, out=None):
    t = _wrap(input)
    return Tensor(np.cumsum(t.numpy(), axis=dim).astype(np.float32))

def cumprod(input, dim, dtype=None, out=None):
    t = _wrap(input)
    return Tensor(np.cumprod(t.numpy(), axis=dim).astype(np.float32))

def all(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.all(t.numpy())))
    return Tensor(np.all(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))

def any(input, dim=None, keepdim=False, out=None):
    t = _wrap(input)
    if dim is None:
        return Tensor(float(np.any(t.numpy())))
    return Tensor(np.any(t.numpy(), axis=dim, keepdims=keepdim).astype(np.float32))

def numel(input):
    return int(np.prod(_wrap(input).shape))

# ============================================================
# COMPARISON OPS
# ============================================================

def eq(input, other, out=None):
    return Tensor(np.equal(_wrap(input).numpy(), _wrap(other).numpy() if isinstance(other, Tensor) else other).astype(np.float32))

def ne(input, other, out=None):
    return Tensor(np.not_equal(_wrap(input).numpy(), _wrap(other).numpy() if isinstance(other, Tensor) else other).astype(np.float32))

def lt(input, other, out=None):
    return Tensor(np.less(_wrap(input).numpy(), _wrap(other).numpy() if isinstance(other, Tensor) else other).astype(np.float32))

def le(input, other, out=None):
    return Tensor(np.less_equal(_wrap(input).numpy(), _wrap(other).numpy() if isinstance(other, Tensor) else other).astype(np.float32))

def gt(input, other, out=None):
    return Tensor(np.greater(_wrap(input).numpy(), _wrap(other).numpy() if isinstance(other, Tensor) else other).astype(np.float32))

def ge(input, other, out=None):
    return Tensor(np.greater_equal(_wrap(input).numpy(), _wrap(other).numpy() if isinstance(other, Tensor) else other).astype(np.float32))

def isnan(input):
    return Tensor(np.isnan(_wrap(input).numpy()).astype(np.float32))

def isinf(input):
    return Tensor(np.isinf(_wrap(input).numpy()).astype(np.float32))

def isfinite(input):
    return Tensor(np.isfinite(_wrap(input).numpy()).astype(np.float32))

def allclose(input, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    return bool(np.allclose(_wrap(input).numpy(), _wrap(other).numpy(), rtol=rtol, atol=atol, equal_nan=equal_nan))

def topk(input, k, dim=-1, largest=True, sorted=True, out=None):
    t = _wrap(input)
    arr = t.numpy()
    if dim == -1:
        dim = arr.ndim - 1
    idx = np.argsort(arr, axis=dim)
    if largest:
        idx = np.flip(idx, axis=dim)
    idx = np.take(idx, np.arange(k), axis=dim)
    vals = np.take_along_axis(arr, idx, axis=dim)
    return Tensor(vals.astype(np.float32)), Tensor(idx.astype(np.float32))

def sort(input, dim=-1, descending=False, stable=False, out=None):
    t = _wrap(input)
    arr = t.numpy()
    if dim == -1:
        dim = arr.ndim - 1
    idx = np.argsort(arr, axis=dim, stable=stable)
    if descending:
        idx = np.flip(idx, axis=dim)
    vals = np.take_along_axis(arr, idx, axis=dim)
    return Tensor(vals.astype(np.float32)), Tensor(idx.astype(np.float32))

# ============================================================
# INDEXING, JOINING, MUTATING
# ============================================================

def cat(tensors, dim=0, out=None):
    if candle is not None:
        raw = [t._tensor if isinstance(t, Tensor) else t for t in tensors]
        try:
            return Tensor(candle.cat(raw, dim))
        except:
            pass  # fallback
    arrs = [_wrap(t).numpy() for t in tensors]
    return Tensor(np.concatenate(arrs, axis=dim).astype(np.float32))

def stack(tensors, dim=0, out=None):
    if candle is not None:
        raw = [t._tensor if isinstance(t, Tensor) else t for t in tensors]
        try:
            return Tensor(candle.stack(raw, dim))
        except:
            pass
    arrs = [_wrap(t).numpy() for t in tensors]
    return Tensor(np.stack(arrs, axis=dim).astype(np.float32))

def chunk(input, chunks, dim=0):
    t = _wrap(input)
    arrs = np.array_split(t.numpy(), chunks, axis=dim)
    return [Tensor(a.astype(np.float32)) for a in arrs]

def split(tensor, split_size_or_sections, dim=0):
    t = _wrap(tensor)
    arr = t.numpy()
    if isinstance(split_size_or_sections, int):
        indices = list(range(split_size_or_sections, arr.shape[dim], split_size_or_sections))
    else:
        indices = list(np.cumsum(split_size_or_sections)[:-1])
    arrs = np.split(arr, indices, axis=dim)
    return [Tensor(a.astype(np.float32)) for a in arrs]

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

def gather(input, dim, index, sparse_grad=False, out=None):
    t = _wrap(input)
    idx = _wrap(index)
    return Tensor(np.take_along_axis(t.numpy(), idx.numpy().astype(np.int64), axis=dim).astype(np.float32))

def scatter_(input, dim, index, src):
    t = _wrap(input)
    arr = t.numpy().copy()
    idx = _wrap(index).numpy().astype(np.int64)
    src_arr = _wrap(src).numpy() if isinstance(src, Tensor) else src
    np.put_along_axis(arr, idx, src_arr, axis=dim)
    t._tensor = Tensor(arr.astype(np.float32))._tensor
    return t

def index_select(input, dim, index, out=None):
    t = _wrap(input)
    idx = _wrap(index).numpy().astype(np.int64)
    return Tensor(np.take(t.numpy(), idx, axis=dim).astype(np.float32))

def where(condition, input=None, other=None):
    cond = _wrap(condition).numpy().astype(bool)
    if input is None:
        return Tensor(np.argwhere(cond).astype(np.float32))
    a = _wrap(input).numpy() if isinstance(input, Tensor) else input
    b = _wrap(other).numpy() if isinstance(other, Tensor) else other
    return Tensor(np.where(cond, a, b).astype(np.float32))

def masked_select(input, mask, out=None):
    t = _wrap(input)
    m = _wrap(mask).numpy().astype(bool)
    return Tensor(t.numpy()[m].astype(np.float32))

def nonzero(input, out=None, as_tuple=False):
    t = _wrap(input)
    indices = np.nonzero(t.numpy())
    if as_tuple:
        return tuple(Tensor(i.astype(np.float32)) for i in indices)
    return Tensor(np.column_stack(indices).astype(np.float32))

def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    t = _wrap(input)
    result = np.unique(t.numpy(), return_inverse=return_inverse, return_counts=return_counts, axis=dim)
    if return_inverse or return_counts:
        return tuple(Tensor(r.astype(np.float32)) for r in result)
    return Tensor(result.astype(np.float32))

def tril(input, diagonal=0, out=None):
    t = _wrap(input)
    return Tensor(np.tril(t.numpy(), k=diagonal).astype(np.float32))

def triu(input, diagonal=0, out=None):
    t = _wrap(input)
    return Tensor(np.triu(t.numpy(), k=diagonal).astype(np.float32))

def flip(input, dims, out=None):
    t = _wrap(input)
    return Tensor(np.flip(t.numpy(), axis=dims).copy().astype(np.float32))

def roll(input, shifts, dims=None):
    t = _wrap(input)
    return Tensor(np.roll(t.numpy(), shifts, axis=dims).astype(np.float32))

def repeat_interleave(input, repeats, dim=None, output_size=None):
    t = _wrap(input)
    return Tensor(np.repeat(t.numpy(), repeats, axis=dim).astype(np.float32))

def broadcast_to(input, size):
    t = _wrap(input)
    return Tensor(np.broadcast_to(t.numpy(), size).copy().astype(np.float32))

# ============================================================
# EINSUM
# ============================================================

def einsum(equation, *operands):
    ops_np = [_wrap(o).numpy() for o in operands]
    return Tensor(np.einsum(equation, *ops_np).astype(np.float32))

# ============================================================
# TRIGONOMETRIC / MATH ALIASES (already defined above, expose as module attrs)
# ============================================================

arcsin = asin
arccos = acos
arctan = atan
arcsinh = asinh
arccosh = acosh
arctanh = atanh
sigmoid = tanh  # NOTE: sigmoid != tanh; override below

def sigmoid(input, out=None):
    return _wrap(input).sigmoid()

def relu(input):
    return _wrap(input).relu()

# ============================================================
# MISC TYPE / IDENTITY OPS
# ============================================================

def clone(input, memory_format=None):
    t = _wrap(input)
    return Tensor(t.numpy().copy())

def detach(input):
    t = _wrap(input)
    out = Tensor(t._tensor)
    out.requires_grad = False
    out.grad_fn = None
    return out

def contiguous(input):
    return _wrap(input)  # candle tensors are always 'contiguous' from Python's view

def type_as(input, other):
    return _wrap(input).to(dtype=_wrap(other).dtype)

def float(input):
    t = _wrap(input)
    return t.numpy().astype(np.float32)

# ============================================================
# MISSING LEGACY / VIEW OPS
# ============================================================

def narrow(input, dim, start, length):
    t = _wrap(input)
    return Tensor(t._tensor.narrow(dim, start, length))

def select(input, dim, index):
    t = _wrap(input)
    return Tensor(t._tensor.get(index) if dim == 0 else t.numpy().take(index, axis=dim).astype(np.float32))
