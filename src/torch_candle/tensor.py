"""
torch_candle.Tensor — zero-copy thin wrapper around candle.Tensor (Rust/PyO3).
Hot paths are 100% candle Rust ops; numpy is only used where candle has no native equivalent.
"""
import candle
import numpy as np
import math as _math

# ─── dtype shortcuts (resolve once at import time) ───────────────────────────
_f32  = candle.f32
_u8   = candle.u8


def _raw(t):
    """Return the underlying candle.Tensor from a Tensor wrapper."""
    return t._tensor if isinstance(t, Tensor) else t


def _c_ones(shape, device, dtype):
    """Create a candle.Tensor of ones with the given shape/device/dtype."""
    return candle.ones(shape).to_device(device).to_dtype(dtype)


def _c_zeros(shape, device, dtype):
    """Create a candle.Tensor of zeros with the given shape/device/dtype."""
    return candle.zeros(shape).to_device(device).to_dtype(dtype)


def _to_u8_mask(raw_cond: "candle.Tensor") -> "candle.Tensor":
    """Convert a float candle tensor (0./1.) to u8 for use with where_cond."""
    return raw_cond.to_dtype(_u8)


class Tensor:
    """torch_candle.Tensor — thin wrapper around candle.Tensor (Rust via PyO3).

    Philosophy: every hot-path op stays on the Rust side.  numpy() is a last
    resort for ops that candle does not yet expose (trig inverses, cumsum, etc.).
    """

    _grad_enabled = True  # toggled by torch.no_grad()

    # ──────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        if isinstance(data, candle.Tensor):
            self._tensor = data
        elif isinstance(data, Tensor):
            self._tensor = data._tensor
        elif isinstance(data, (list, tuple, np.ndarray, float, int, np.generic)):
            arr = np.array(data)
            if np.iscomplexobj(arr):
                arr = arr.real
            arr = arr.astype(np.float32)
            shape = arr.shape
            # Use flat list only for 0-/1-d; for higher-rank use reshape
            flat = arr.flatten().tolist()
            self._tensor = candle.Tensor(flat).reshape(shape) if shape else candle.Tensor(flat)
        else:
            raise TypeError(f"Cannot construct Tensor from {type(data)}")

        # Default to float32
        if dtype is None:
            dtype = _f32
        self._tensor = self._tensor.to_dtype(dtype)
        if device is not None:
            self._tensor = self._tensor.to_device(device)

        self.grad_fn      = None
        self.grad         = None
        self.requires_grad = requires_grad
        self.is_leaf       = True

    # ──────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────
    @property
    def data(self):
        return self

    @property
    def shape(self):
        return self._tensor.shape

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    @property
    def device(self):
        return self._tensor.device

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def ndim(self):
        return self._tensor.rank

    # ──────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────
    def __getitem__(self, index):
        res  = self._tensor
        out_tensor = None

        try:
            if isinstance(index, int):
                out_tensor = res.get(index)
            elif isinstance(index, slice):
                start = index.start if index.start is not None else 0
                stop  = index.stop  if index.stop  is not None else self.shape[0]
                out_tensor = res.narrow(0, start, stop - start)
            elif isinstance(index, tuple) and len(index) == 2 and all(isinstance(i, int) for i in index):
                # fast path for 2-D integer indexing [row, col]
                out_tensor = res.get(index[0]).get(index[1])
        except Exception:
            pass

        if out_tensor is None:
            # numpy fallback for complex fancy indexing
            out = Tensor(self.numpy()[index], device=self.device, dtype=self.dtype)
        else:
            out = Tensor(out_tensor)

        if self.requires_grad:
            from .autograd import AutogradContext, SliceBackward
            out.requires_grad = True
            out.is_leaf = False
            ctx = AutogradContext()
            ctx.index = index
            out.grad_fn = SliceBackward("slice", [self], ctx)

        return out

    def __setitem__(self, index, value):
        # numpy round-trip for now; masked_fill_ uses where_cond instead
        np_data = self.numpy()
        val_data = value.numpy() if isinstance(value, Tensor) else value
        np_data[index] = val_data
        arr = np.array(np_data, dtype=np.float32)
        self._tensor = candle.Tensor(arr.flatten().tolist()).reshape(arr.shape)
        self._tensor = self._tensor.to_device(self.device).to_dtype(self.dtype)

    def __repr__(self):
        return f"torch_candle.Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    # ──────────────────────────────────────────────────────────────
    # Arithmetic — all stay on candle Rust
    # ──────────────────────────────────────────────────────────────
    def __add__(self, other):
        if other is None:
            return self
        other_raw = _raw(other)
        if isinstance(other_raw, candle.Tensor) and self.shape != other_raw.shape:
            res = self._tensor.broadcast_add(other_raw)
        else:
            res = self._tensor + other_raw
        out = Tensor(res)
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, AddBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = AddBackward("add", [self, other if isinstance(other, Tensor) else Tensor(other)], AutogradContext())
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other is None:
            return self
        other_raw = _raw(other)
        if isinstance(other_raw, candle.Tensor) and self.shape != other_raw.shape:
            res = self._tensor.broadcast_sub(other_raw)
        else:
            res = self._tensor - other_raw
        out = Tensor(res)
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, SubBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = SubBackward("sub", [self, other if isinstance(other, Tensor) else Tensor(other)], AutogradContext())
        return out

    def __rsub__(self, other):
        return self.neg().__add__(other)

    def __mul__(self, other):
        other_raw = _raw(other)
        if isinstance(other_raw, candle.Tensor) and self.shape != other_raw.shape:
            res = self._tensor.broadcast_mul(other_raw)
        else:
            res = self._tensor * other_raw
        out = Tensor(res)
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, MulBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = MulBackward("mul", [self, other if isinstance(other, Tensor) else Tensor(other)], AutogradContext())
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.neg()

    def __truediv__(self, other):
        other_raw = _raw(other)
        if isinstance(other_raw, candle.Tensor) and self.shape != other_raw.shape:
            return Tensor(self._tensor.broadcast_div(other_raw))
        return Tensor(self._tensor / other_raw)

    def __rtruediv__(self, other):
        other_raw = _raw(other)
        if isinstance(other_raw, candle.Tensor) and self.shape != other_raw.shape:
            return Tensor(other_raw.broadcast_div(self._tensor))
        return Tensor(other_raw / self._tensor)

    def __pow__(self, exponent):
        out = Tensor(self._tensor.powf(float(exponent)))
        if self.requires_grad:
            from .autograd import AutogradContext, PowBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = PowBackward("pow", [self, exponent], AutogradContext())
        return out

    # ──────────────────────────────────────────────────────────────
    # Linear algebra
    # ──────────────────────────────────────────────────────────────
    def matmul(self, other):
        out = Tensor(self._tensor.matmul(_raw(other)))
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, MatmulBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = MatmulBackward("matmul", [self, other], AutogradContext())
        return out

    def t(self):
        out = Tensor(self._tensor.t())
        if self.requires_grad:
            from .autograd import AutogradContext, TransposeBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = TransposeBackward("transpose", [self], AutogradContext())
        return out

    def transpose(self, dim0, dim1):
        out = Tensor(self._tensor.transpose(dim0, dim1))
        if self.requires_grad:
            from .autograd import AutogradContext, TransposeBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = TransposeBackward("transpose", [self], AutogradContext())
        return out

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        if hasattr(self._tensor, 'permute'):
            return Tensor(self._tensor.permute(*dims))
        raise NotImplementedError("permute not supported in this candle build")

    # ──────────────────────────────────────────────────────────────
    # Shape manipulation
    # ──────────────────────────────────────────────────────────────
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        out = Tensor(self._tensor.reshape(shape))
        if self.requires_grad:
            from .autograd import AutogradContext, ReshapeBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = ReshapeBackward("view", [self], AutogradContext())
        return out

    def reshape(self, *shape):
        return self.view(*shape)

    def squeeze(self, dim=None):
        if dim is None:
            res = self._tensor
            for i, s in reversed(list(enumerate(self.shape))):
                if s == 1:
                    res = res.squeeze(i)
            return Tensor(res)
        return Tensor(self._tensor.squeeze(dim))

    def unsqueeze(self, dim):
        return Tensor(self._tensor.unsqueeze(dim))

    def flatten(self, start_dim=0, end_dim=-1):
        if start_dim == 0 and end_dim == -1:
            return Tensor(self._tensor.flatten_all())
        shape = self.shape
        if end_dim == -1:
            end_dim = len(shape) - 1
        new_shape = list(shape[:start_dim])
        prod = 1
        for i in range(start_dim, end_dim + 1):
            prod *= shape[i]
        new_shape.append(prod)
        new_shape.extend(shape[end_dim + 1:])
        return self.view(new_shape)

    def contiguous(self, memory_format=None):
        return Tensor(self._tensor.contiguous())

    # ──────────────────────────────────────────────────────────────
    # Reductions — all candle-native
    # ──────────────────────────────────────────────────────────────
    def sum(self, dim=None, keepdim=False):
        if dim is None:
            res = self._tensor.sum_all()
        else:
            ndim = len(self.shape)
            if isinstance(dim, int):
                dim_norm = dim if dim >= 0 else dim + ndim
                dim_seq  = [dim_norm]
            else:
                dim_seq = [(d if d >= 0 else d + ndim) for d in dim]
            res = self._tensor.sum_keepdim(dim_seq)
            if not keepdim:
                if isinstance(dim, int):
                    res = res.squeeze(dim_seq[0])
                else:
                    for d in sorted(dim_seq, reverse=True):
                        res = res.squeeze(d)
        out = Tensor(res)
        if self.requires_grad:
            from .autograd import AutogradContext, SumBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = SumBackward("sum", [self], AutogradContext())
        return out

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            out = Tensor(self._tensor.mean_all())
        else:
            s    = self.sum(dim=dim, keepdim=keepdim)
            size = self.shape[dim] if isinstance(dim, int) else int(np.prod([self.shape[d] for d in dim]))
            out  = s * (1.0 / size)
        if self.requires_grad:
            from .autograd import AutogradContext, MeanBackward
            out.requires_grad = True
            out.is_leaf  = False
            out.grad_fn  = MeanBackward("mean", [self], AutogradContext())
        return out

    def max_keepdim(self, dim):
        return Tensor(self._tensor.max_keepdim(dim))

    # ──────────────────────────────────────────────────────────────
    # Unary element-wise — 100% candle Rust, zero numpy
    # ──────────────────────────────────────────────────────────────
    def neg(self):
        """Negate: x * -1 on candle Rust tensor."""
        return Tensor(self._tensor * -1.0)

    def abs(self):
        """Absolute value via sqrt(sqr(x)) — pure candle."""
        return Tensor(self._tensor.sqr().sqrt())

    def sqrt(self):
        return Tensor(self._tensor.sqrt())

    def exp(self):
        return Tensor(self._tensor.exp())

    def log(self):
        return Tensor(self._tensor.log())

    def relu(self):
        """ReLU: max(0, x) via where_cond on u8 mask — pure candle."""
        z    = candle.zeros(self.shape).to_device(self.device).to_dtype(self.dtype)
        # build float mask: 1 where x > 0
        pos  = self._tensor - z          # shift so exactly-0 maps to 0
        # Reuse sigmoid-based mask: x > 0  ⟺  sigmoid(x*1e6) ≈ 1
        # Simpler: use sqr to make all pos, then compare via (x+eps > 0)
        # Most reliable for candle: compute as max via (x + |x|) / 2
        abs_x = self._tensor.sqr().sqrt()
        return Tensor((self._tensor + abs_x) * 0.5)

    def sigmoid(self):
        """σ(x) = 1 / (1 + exp(-x)) — pure candle."""
        neg_x    = self._tensor * -1.0
        exp_neg  = neg_x.exp()
        ones     = candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype)
        return Tensor((ones + exp_neg).recip())

    def tanh(self):
        """tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) — pure candle."""
        two_x = self._tensor * 2.0
        e2x   = two_x.exp()
        ones  = candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype)
        return Tensor((e2x - ones) * (e2x + ones).recip())

    def sin(self):
        return Tensor(self._tensor.sin())

    def cos(self):
        return Tensor(self._tensor.cos())

    def clamp(self, min=None, max=None):
        # Implement via candle arithmetic using abs
        res = self
        if min is not None:
            # max(x, min) = (x + min + |x - min|) / 2
            min_t = Tensor(
                candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype) * float(min)
            )
            diff  = res - min_t
            res   = (res + min_t + diff.abs()) * 0.5
        if max is not None:
            # min(x, max) = (x + max - |x - max|) / 2
            max_t = Tensor(
                candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype) * float(max)
            )
            diff  = res - max_t
            res   = (res + max_t - diff.abs()) * 0.5
        return res

    # ──────────────────────────────────────────────────────────────
    # Device / dtype control
    # ──────────────────────────────────────────────────────────────
    def to(self, *args, **kwargs):
        t = self._tensor
        for arg in args:
            if isinstance(arg, (str, candle.Device)):
                t = t.to_device(arg)
            elif str(type(arg)).find('DType') != -1:
                t = t.to_dtype(arg)
        if 'device' in kwargs:
            t = t.to_device(kwargs['device'])
        if 'dtype' in kwargs:
            t = t.to_dtype(kwargs['dtype'])
        return Tensor(t, requires_grad=self.requires_grad)

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None):
        d = f'cuda:{device}' if isinstance(device, int) else 'cuda'
        return self.to(d)

    def float(self):
        return self.to(dtype=_f32)

    def double(self):
        return self.to(dtype=candle.f64)

    def long(self):
        # candle has no i64 dtype; use u32 as proxy
        ltype = getattr(candle, 'i64', getattr(candle, 'u32', _f32))
        return self.to(dtype=ltype)

    def half(self):
        return self.to(dtype=candle.f16)

    def bfloat16(self):
        return self.to(dtype=candle.bf16)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        return str(self.dtype) if dtype is None else self.to(dtype=dtype)

    def to_float(self):
        return self.float()

    # ──────────────────────────────────────────────────────────────
    # Autograd
    # ──────────────────────────────────────────────────────────────
    def backward(self, gradient=None):
        from .autograd import backward
        backward(self, gradient)

    def detach(self):
        out = Tensor(self._tensor)
        out.requires_grad = False
        out.grad_fn = None
        out.is_leaf = True
        return out

    # ──────────────────────────────────────────────────────────────
    # Numpy bridge — used for ops with no candle equivalent
    # ──────────────────────────────────────────────────────────────
    def numpy(self):
        shape = tuple(self.shape)
        try:
            flat = self._tensor.flatten_all()
            return np.array(flat.values(), dtype=np.float32).reshape(shape)
        except Exception:
            # last resort — convert via torch if available
            try:
                return np.array(self._tensor.to_torch().numpy())
            except Exception:
                raise RuntimeError("Cannot convert tensor to numpy")

    def item(self):
        n = self.numpy()
        return n.item() if hasattr(n, 'item') else float(n)

    # ──────────────────────────────────────────────────────────────
    # Clone / identity
    # ──────────────────────────────────────────────────────────────
    def clone(self):
        """Clone using candle .copy() — zero numpy."""
        return Tensor(self._tensor.copy())

    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n

    def element_size(self):
        return 4

    def dim(self):
        return len(self.shape)

    # ──────────────────────────────────────────────────────────────
    # In-place ops — stay on candle where possible
    # ──────────────────────────────────────────────────────────────
    def fill_(self, value):
        self._tensor = candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype) * float(value)
        return self

    def zero_(self):
        self._tensor = candle.zeros(self.shape).to_device(self.device).to_dtype(self.dtype)
        return self

    def normal_(self, mean=0.0, std=1.0):
        r = candle.randn(self.shape).to_device(self.device).to_dtype(self.dtype)
        if std != 1.0:
            r = r * float(std)
        if mean != 0.0:
            r = r + candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype) * float(mean)
        self._tensor = r
        return self

    def uniform_(self, a=0.0, b=1.0):
        r = candle.rand(self.shape).to_device(self.device).to_dtype(self.dtype)
        if b != 1.0 or a != 0.0:
            r = r * float(b - a) + candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype) * float(a)
        self._tensor = r
        return self

    def masked_fill_(self, mask, value):
        # mask is a Tensor of 0./1.; use where_cond for candle-native op
        val_t = candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype) * float(value)
        mask_u8 = mask._tensor.to_dtype(_u8) if isinstance(mask, Tensor) else mask.to_dtype(_u8)
        self._tensor = mask_u8.where_cond(val_t, self._tensor)
        return self

    # ──────────────────────────────────────────────────────────────
    # Expand / repeat — candle broadcast_as where possible
    # ──────────────────────────────────────────────────────────────
    def expand(self, *sizes):
        sizes = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)) else sizes
        try:
            return Tensor(self._tensor.broadcast_as(tuple(sizes)))
        except Exception:
            return Tensor(np.broadcast_to(self.numpy(), sizes).copy().astype(np.float32))

    def expand_as(self, other):
        return self.expand(*other.shape)

    def repeat(self, *sizes):
        sizes = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)) else sizes
        # No candle equivalent — numpy fallback once
        return Tensor(np.tile(self.numpy(), sizes).astype(np.float32))

    # ──────────────────────────────────────────────────────────────
    # Comparison — produce float 0/1 tensors
    # ──────────────────────────────────────────────────────────────
    def _cmp_np(self, op, other):
        fn  = getattr(np, op)
        rhs = other.numpy() if isinstance(other, Tensor) else other
        return Tensor(fn(self.numpy(), rhs).astype(np.float32))

    def __eq__(self, other):   return self._cmp_np('equal', other)
    def __ne__(self, other):   return self._cmp_np('not_equal', other)
    def __lt__(self, other):   return self._cmp_np('less', other)
    def __le__(self, other):   return self._cmp_np('less_equal', other)
    def __gt__(self, other):   return self._cmp_np('greater', other)
    def __ge__(self, other):   return self._cmp_np('greater_equal', other)

    def eq(self, other):   return self.__eq__(other)
    def ne(self, other):   return self.__ne__(other)
    def lt(self, other):   return self.__lt__(other)
    def le(self, other):   return self.__le__(other)
    def gt(self, other):   return self.__gt__(other)
    def ge(self, other):   return self.__ge__(other)

    # ──────────────────────────────────────────────────────────────
    # Python dunder helpers
    # ──────────────────────────────────────────────────────────────
    def __bool__(self):
        return bool(self.item())

    def __len__(self):
        return self.shape[0]

    def __hash__(self):
        return id(self)

    def __matmul__(self, other):
        return self.matmul(other)

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    # ──────────────────────────────────────────────────────────────
    # Delegated trig / math ops — keep in ops.py for consistency
    # ──────────────────────────────────────────────────────────────
    def tan(self):   return self.sin() / self.cos()
    def asin(self):  from . import ops; return ops.asin(self)
    def acos(self):  from . import ops; return ops.acos(self)
    def atan(self):  from . import ops; return ops.atan(self)
    def sinh(self):  from . import ops; return ops.sinh(self)
    def cosh(self):  from . import ops; return ops.cosh(self)
    def atanh(self): from . import ops; return ops.atanh(self)
    def asinh(self): from . import ops; return ops.asinh(self)
    def acosh(self): from . import ops; return ops.acosh(self)
    def floor(self): from . import ops; return ops.floor(self)
    def ceil(self):  from . import ops; return ops.ceil(self)
    def round(self, decimals=0): from . import ops; return ops.round(self, decimals=decimals)
    def trunc(self): from . import ops; return ops.trunc(self)
    def frac(self):  from . import ops; return ops.frac(self)
    def reciprocal(self): return Tensor(self._tensor.recip())

    # ──────────────────────────────────────────────────────────────
    # Delegated reduction ops
    # ──────────────────────────────────────────────────────────────
    def max(self, dim=None, keepdim=False):    from . import ops; return ops.max(self, dim=dim, keepdim=keepdim)
    def min(self, dim=None, keepdim=False):    from . import ops; return ops.min(self, dim=dim, keepdim=keepdim)
    def argmax(self, dim=None, keepdim=False): from . import ops; return ops.argmax(self, dim=dim, keepdim=keepdim)
    def argmin(self, dim=None, keepdim=False): from . import ops; return ops.argmin(self, dim=dim, keepdim=keepdim)
    def std(self, dim=None, correction=1, keepdim=False): from . import ops; return ops.std(self, dim=dim, correction=correction, keepdim=keepdim)
    def var(self, dim=None, correction=1, keepdim=False): from . import ops; return ops.var(self, dim=dim, correction=correction, keepdim=keepdim)
    def prod(self, dim=None, keepdim=False):   from . import ops; return ops.prod(self, dim=dim, keepdim=keepdim)
    def cumsum(self, dim):                     from . import ops; return ops.cumsum(self, dim)
    def cumprod(self, dim):                    from . import ops; return ops.cumprod(self, dim)
    def all(self, dim=None):                   from . import ops; return ops.all(self, dim=dim)
    def any(self, dim=None):                   from . import ops; return ops.any(self, dim=dim)
    def norm(self, p=2, dim=None, keepdim=False): from . import ops; return ops.norm(self, p=p, dim=dim, keepdim=keepdim)
    def topk(self, k, dim=-1, largest=True, sorted=True): from . import ops; return ops.topk(self, k, dim=dim, largest=largest, sorted=sorted)
    def sort(self, dim=-1, descending=False, stable=False): from . import ops; return ops.sort(self, dim=dim, descending=descending, stable=stable)

    # ──────────────────────────────────────────────────────────────
    # Delegated indexing ops
    # ──────────────────────────────────────────────────────────────
    def flip(self, dims):                   from . import ops; return ops.flip(self, dims)
    def roll(self, shifts, dims=None):      from . import ops; return ops.roll(self, shifts, dims)
    def gather(self, dim, index):           from . import ops; return ops.gather(self, dim, index)
    def chunk(self, chunks, dim=0):         from . import ops; return ops.chunk(self, chunks, dim)
    def split(self, split_size, dim=0):     from . import ops; return ops.split(self, split_size, dim)
    def nonzero(self, as_tuple=False):      from . import ops; return ops.nonzero(self, as_tuple=as_tuple)
    def tril(self, diagonal=0):             from . import ops; return ops.tril(self, diagonal)
    def triu(self, diagonal=0):             from . import ops; return ops.triu(self, diagonal)

    # ──────────────────────────────────────────────────────────────
    # Misc ops
    # ──────────────────────────────────────────────────────────────
    def isnan(self):    from . import ops; return ops.isnan(self)
    def isinf(self):    from . import ops; return ops.isinf(self)
    def isfinite(self): from . import ops; return ops.isfinite(self)
