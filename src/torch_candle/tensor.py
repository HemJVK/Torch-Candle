try:
    import candle
except ImportError:
    candle = None
import numpy as np

class Tensor:
    """torch_candle.Tensor — thin wrapper around candle.Tensor (Rust via PyO3)."""
    _grad_enabled = True  # toggled by torch.no_grad()

    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        if candle is not None and isinstance(data, candle.Tensor):
            self._tensor = data
        elif isinstance(data, (list, tuple, np.ndarray, float, int, np.generic)):
            if candle is None: raise ImportError("candle not found")
            # Unified approach for candle 0.2.1: convert to flat list + reshape
            arr = np.array(data)
            if np.iscomplexobj(arr):
                arr = arr.real.astype(np.float32)
            shape = arr.shape
            flat_data = arr.flatten().tolist()
            if candle is not None:
                self._tensor = candle.Tensor(flat_data).reshape(shape)
            else:
                self._tensor = None # Should have raised earlier
        elif isinstance(data, Tensor):
            self._tensor = data._tensor
        else:
            raise TypeError(f"Cannot construct Tensor from {type(data)}")

        if dtype is None:
            # Default to float32 for numeric data
            try:
                from . import float32
                dtype = float32
            except ImportError:
                dtype = None
            
        if dtype is not None:
            self._tensor = self._tensor.to_dtype(dtype)
        if device is not None:
            self._tensor = self._tensor.to_device(device)
            
        self.grad_fn = None
        self.grad = None
        self.requires_grad = requires_grad
        self.is_leaf = True

    @property
    def data(self):
        return self

    @property
    def shape(self):
        return self._tensor.shape

    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def device(self):
        return self._tensor.device

    @property
    def dtype(self):
        return self._tensor.dtype

    @property
    def ndim(self):
        return self._tensor.rank

    def __getitem__(self, index):
        res = self._tensor
        out_tensor = None
        
        # Try native candle indexing first for performance
        try:
            if isinstance(index, int):
                out_tensor = res.get(index)
            elif isinstance(index, slice):
                start = index.start if index.start is not None else 0
                stop = index.stop if index.stop is not None else self.shape[0]
                out_tensor = res.narrow(0, start, stop - start)
        except Exception:
            pass
            
        if out_tensor is None:
            # Robust fallback using numpy
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
        # Slice assignment: this is a MUST HAVE
        # We convert to numpy, set, and convert back
        # Note: This makes the operation non-differentiable in the current autograd placeholder
        np_data = self.numpy()
        val_data = value.numpy() if isinstance(value, Tensor) else value
        np_data[index] = val_data
        
        # Re-initialize _tensor
        arr = np.array(np_data)
        shape = arr.shape
        flat_data = arr.flatten().tolist()
        self._tensor = candle.Tensor(flat_data).reshape(shape).to_device(self.device).to_dtype(self.dtype)

    def __repr__(self):
        return f"torch_candle.Tensor({self._tensor.__repr__()}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        if other is None:
            return self
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        res_data = self._tensor.broadcast_add(other_tensor) if (isinstance(other_tensor, candle.Tensor) and self.shape != other_tensor.shape) else (self._tensor + other_tensor)
        out = Tensor(res_data)
        
        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, AddBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = AddBackward("add", [self, other if isinstance(other, Tensor) else Tensor(other)], AutogradContext())
            
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other is None:
            return self
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        res_data = self._tensor.broadcast_sub(other_tensor) if (isinstance(other_tensor, candle.Tensor) and self.shape != other_tensor.shape) else (self._tensor - other_tensor)
        out = Tensor(res_data)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, SubBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = SubBackward("sub", [self, other if isinstance(other, Tensor) else Tensor(other)], AutogradContext())
            
        return out

    def __rsub__(self, other):
        # other - self = other + (-self)
        neg_self = self.__mul__(-1.0)
        res = neg_self.__add__(other)
        return res

    def __mul__(self, other):
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        res_data = self._tensor.broadcast_mul(other_tensor) if (isinstance(other_tensor, candle.Tensor) and self.shape != other_tensor.shape) else (self._tensor * other_tensor)
        out = Tensor(res_data)

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            from .autograd import AutogradContext, MulBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = MulBackward("mul", [self, other if isinstance(other, Tensor) else Tensor(other)], AutogradContext())
            
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __truediv__(self, other):
        other_tensor = other._tensor if hasattr(other, '_tensor') else other
        if candle is not None and isinstance(other_tensor, candle.Tensor) and self.shape != other_tensor.shape:
             res_data = self._tensor.broadcast_div(other_tensor)
        else:
             res_data = self._tensor / other_tensor
        return Tensor(res_data)

    def __rtruediv__(self, other):
        other_tensor = other._tensor if hasattr(other, '_tensor') else other
        if candle is not None and isinstance(other_tensor, candle.Tensor) and self.shape != other_tensor.shape:
            return Tensor(other_tensor.broadcast_div(self._tensor))
        return Tensor(other_tensor / self._tensor)

    def neg(self):
        if candle is not None and hasattr(self._tensor, 'neg'):
            return Tensor(self._tensor.neg())
        # Fallback using 0 - x
        return self.__mul__(-1.0)

    def __pow__(self, exponent):
        out = Tensor(self._tensor.powf(exponent))
        if self.requires_grad:
            from .autograd import AutogradContext, PowBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = PowBackward("pow", [self, exponent], AutogradContext())
        return out

    def matmul(self, other):
        out = Tensor(self._tensor.matmul(other._tensor))
        if self.requires_grad or other.requires_grad:
            from .autograd import AutogradContext, MatmulBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = MatmulBackward("matmul", [self, other], AutogradContext())
        return out

    def t(self):
        out = Tensor(self._tensor.t())
        if self.requires_grad:
            from .autograd import AutogradContext, TransposeBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = TransposeBackward("transpose", [self], AutogradContext())
        return out

    def transpose(self, dim0, dim1):
        out = Tensor(self._tensor.transpose(dim0, dim1))
        if self.requires_grad:
            from .autograd import AutogradContext, TransposeBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = TransposeBackward("transpose", [self], AutogradContext())
        return out

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        # simplified permute using multiple transposes if needed
        # but candle might have it if we enable more features
        if candle is not None and hasattr(self._tensor, 'permute'):
            return Tensor(self._tensor.permute(*dims))
        raise NotImplementedError("permute not natively supported and complex fallback not yet implemented")

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        outArr = self._tensor.reshape(shape)
        out = Tensor(outArr)
        if self.requires_grad:
            from .autograd import AutogradContext, ReshapeBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = ReshapeBackward("view", [self], AutogradContext())
        return out

    def reshape(self, *shape):
        return self.view(*shape)

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            res_data = self._tensor.sum_all()
        else:
            ndim = len(self.shape)
            if isinstance(dim, int):
                dim_norm = dim if dim >= 0 else dim + ndim
                dim_seq = [dim_norm]
            else:
                dim_seq = [(d if d >= 0 else d + ndim) for d in dim]
            res_data = self._tensor.sum_keepdim(dim_seq)
            if not keepdim:
                if isinstance(dim, int):
                    res_data = res_data.squeeze(dim_norm)
                else:
                    for d in sorted(dim_seq, reverse=True):
                        res_data = res_data.squeeze(d)
        
        out = Tensor(res_data)
        if hasattr(self, 'requires_grad') and self.requires_grad:
            from .autograd import AutogradContext, SumBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = SumBackward("sum", [self], AutogradContext())
        return out

    def mean(self, dim=None, keepdim=False):
        s = self.sum(dim=dim, keepdim=keepdim)
        if dim is None:
            size = self._tensor.nelements if hasattr(self._tensor, 'nelements') else np.prod(self.shape)
        elif isinstance(dim, int):
            size = self.shape[dim]
        else:
            size = 1
            for d in dim: size *= self.shape[d]
        
        out = s * (1.0 / size)
        if self.requires_grad:
            from .autograd import AutogradContext, MeanBackward
            out.requires_grad = True
            out.is_leaf = False
            out.grad_fn = MeanBackward("mean", [self], AutogradContext())
        return out

    def max_keepdim(self, dim):
        return Tensor(self._tensor.max_keepdim(dim))

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
        if end_dim == -1: end_dim = len(shape) - 1
        new_shape = list(shape[:start_dim])
        prod = 1
        for i in range(start_dim, end_dim + 1):
            prod *= shape[i]
        new_shape.append(prod)
        new_shape.extend(shape[end_dim + 1:])
        return self.view(new_shape)

    def relu(self):
        if hasattr(self._tensor, 'relu'):
            return Tensor(self._tensor.relu())
        return (self + self.abs()) * 0.5

    def abs(self):
        if hasattr(self._tensor, 'abs'):
            return Tensor(self._tensor.abs())
        return Tensor(self._tensor.sqr().sqrt())

    def sigmoid(self):
        if hasattr(self._tensor, 'sigmoid'):
            return Tensor(self._tensor.sigmoid())
        one = candle.ones(self.shape).to_device(self.device).to_dtype(self.dtype)
        denom = one + (self.neg().exp()._tensor)
        return Tensor(denom.recip())

    def tanh(self):
        if hasattr(self._tensor, 'tanh'):
            return Tensor(self._tensor.tanh())
        ep = self.exp()
        en = self.neg().exp()
        return (ep - en) / (ep + en)

    def exp(self):
        return Tensor(self._tensor.exp())

    def log(self):
        return Tensor(self._tensor.log())

    def sqrt(self):
        return Tensor(self._tensor.sqrt())

    def clamp(self, min=None, max=None):
        res = self._tensor
        if min is not None:
            # use absolute logic or something if broadcast_maximum missing
            if hasattr(res, 'broadcast_maximum'):
                res = res.broadcast_maximum(candle.Tensor([min]).to_device(self.device).to_dtype(self.dtype))
        if max is not None:
            if hasattr(res, 'broadcast_minimum'):
                res = res.broadcast_minimum(candle.Tensor([max]).to_device(self.device).to_dtype(self.dtype))
        return Tensor(res)

    def to(self, *args, **kwargs):
        t = self._tensor
        for arg in args:
            if isinstance(arg, (str, candle.Device)): t = t.to_device(arg)
            elif str(type(arg)).find('DType') != -1: t = t.to_dtype(arg)
        if 'device' in kwargs: t = t.to_device(kwargs['device'])
        if 'dtype' in kwargs: t = t.to_dtype(kwargs['dtype'])
        return Tensor(t, requires_grad=self.requires_grad)

    def backward(self, gradient=None):
        from .autograd import backward
        backward(self, gradient)

    def numpy(self):
        # Fallback through torch if possible, or values()
        try:
            return np.array(self._tensor.to_torch().numpy())
        except:
            # Pyo3 limit: values() only implemented up to rank 3
            # So we flatten to 1D, get values, then reshape in numpy
            shape = tuple(self.shape)
            flat = self._tensor.flatten_all()
            return np.array(flat.values()).reshape(shape)

    def item(self):
        n = self.numpy()
        return n.item() if hasattr(n, 'item') else n

    # --- Type helpers ---
    def to_float(self):
        try:
            import candle; return self.to(dtype=getattr(candle, 'f32', None))
        except Exception:
            return Tensor(self.numpy().astype(import_np().float32))

    def float(self): return self.to_float()
    def double(self):
        try:
            import candle; return self.to(dtype=getattr(candle, 'f64', None))
        except Exception:
            return Tensor(self.numpy().astype(np.float64))
    def long(self):
        try:
            import candle; return self.to(dtype=getattr(candle, 'i64', None))
        except Exception:
            return Tensor(self.numpy().astype(np.int64))
    def half(self):
        try:
            import candle; return self.to(dtype=getattr(candle, 'f16', None))
        except Exception:
            return self

    # --- Comparison dunders ---
    def __eq__(self, other):
        return Tensor(np.equal(self.numpy(), other.numpy() if isinstance(other, Tensor) else other).astype(np.float32))
    def __ne__(self, other):
        return Tensor(np.not_equal(self.numpy(), other.numpy() if isinstance(other, Tensor) else other).astype(np.float32))
    def __lt__(self, other):
        return Tensor(np.less(self.numpy(), other.numpy() if isinstance(other, Tensor) else other).astype(np.float32))
    def __le__(self, other):
        return Tensor(np.less_equal(self.numpy(), other.numpy() if isinstance(other, Tensor) else other).astype(np.float32))
    def __gt__(self, other):
        return Tensor(np.greater(self.numpy(), other.numpy() if isinstance(other, Tensor) else other).astype(np.float32))
    def __ge__(self, other):
        return Tensor(np.greater_equal(self.numpy(), other.numpy() if isinstance(other, Tensor) else other).astype(np.float32))
    def __bool__(self): return bool(self.item())
    def __len__(self): return self.shape[0]
    def __hash__(self): return id(self)
    def __matmul__(self, other): return self.matmul(other)
    def __iter__(self):
        for i in range(self.shape[0]): yield self[i]

    # --- Shape/identity ---
    def numel(self): return int(np.prod(self.shape))
    def element_size(self): return 4
    def dim(self): return len(self.shape)
    def clone(self): return Tensor(self.numpy().copy())
    def contiguous(self, memory_format=None): return self
    def detach(self):
        out = Tensor(self._tensor); out.requires_grad = False; out.grad_fn = None; out.is_leaf = True; return out
    def cpu(self): return self.to('cpu')
    def cuda(self, device=None):
        d = f'cuda:{device}' if isinstance(device, int) else 'cuda'; return self.to(d)
    def type(self, dtype=None, non_blocking=False, **kwargs):
        return str(self.dtype) if dtype is None else self.to(dtype=dtype)

    # --- In-place ops ---
    def fill_(self, value):
        arr = np.full(self.shape, value, dtype=np.float32)
        self._tensor = Tensor(arr.tolist(), device=self.device, dtype=self.dtype)._tensor; return self
    def zero_(self): return self.fill_(0.0)
    def normal_(self, mean=0.0, std=1.0):
        arr = np.random.normal(mean, std, self.shape).astype(np.float32)
        self._tensor = Tensor(arr.tolist(), device=self.device, dtype=self.dtype)._tensor; return self
    def uniform_(self, a=0.0, b=1.0):
        arr = np.random.uniform(a, b, self.shape).astype(np.float32)
        self._tensor = Tensor(arr.tolist(), device=self.device, dtype=self.dtype)._tensor; return self
    def masked_fill_(self, mask, value):
        arr = self.numpy().copy()
        arr[mask.numpy().astype(bool)] = value
        self._tensor = Tensor(arr.astype(np.float32), device=self.device, dtype=self.dtype)._tensor; return self

    # --- Expand/repeat ---
    def repeat(self, *sizes):
        sizes = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)) else sizes
        return Tensor(np.tile(self.numpy(), sizes).astype(np.float32))
    def expand(self, *sizes):
        sizes = sizes[0] if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)) else sizes
        return Tensor(np.broadcast_to(self.numpy(), sizes).copy().astype(np.float32))
    def expand_as(self, other): return self.expand(*other.shape)

    # --- Trig/math delegation to ops ---
    def sin(self):      from . import ops; return ops.sin(self)
    def cos(self):      from . import ops; return ops.cos(self)
    def tan(self):      from . import ops; return ops.tan(self)
    def asin(self):     from . import ops; return ops.asin(self)
    def acos(self):     from . import ops; return ops.acos(self)
    def atan(self):     from . import ops; return ops.atan(self)
    def sinh(self):     from . import ops; return ops.sinh(self)
    def cosh(self):     from . import ops; return ops.cosh(self)
    def atanh(self):    from . import ops; return ops.atanh(self)
    def asinh(self):    from . import ops; return ops.asinh(self)
    def acosh(self):    from . import ops; return ops.acosh(self)
    def floor(self):    from . import ops; return ops.floor(self)
    def ceil(self):     from . import ops; return ops.ceil(self)
    def round(self, decimals=0): from . import ops; return ops.round(self, decimals=decimals)
    def trunc(self):    from . import ops; return ops.trunc(self)
    def frac(self):     from . import ops; return ops.frac(self)
    def reciprocal(self): from . import ops; return ops.reciprocal(self)
    def neg(self):
        if candle and hasattr(self._tensor, 'neg'): return Tensor(self._tensor.neg())
        return self * -1.0

    # --- Reduction delegation ---
    def max(self, dim=None, keepdim=False):  from . import ops; return ops.max(self, dim=dim, keepdim=keepdim)
    def min(self, dim=None, keepdim=False):  from . import ops; return ops.min(self, dim=dim, keepdim=keepdim)
    def argmax(self, dim=None, keepdim=False): from . import ops; return ops.argmax(self, dim=dim, keepdim=keepdim)
    def argmin(self, dim=None, keepdim=False): from . import ops; return ops.argmin(self, dim=dim, keepdim=keepdim)
    def std(self, dim=None, correction=1, keepdim=False): from . import ops; return ops.std(self, dim=dim, correction=correction, keepdim=keepdim)
    def var(self, dim=None, correction=1, keepdim=False): from . import ops; return ops.var(self, dim=dim, correction=correction, keepdim=keepdim)
    def prod(self, dim=None, keepdim=False): from . import ops; return ops.prod(self, dim=dim, keepdim=keepdim)
    def cumsum(self, dim): from . import ops; return ops.cumsum(self, dim)
    def cumprod(self, dim): from . import ops; return ops.cumprod(self, dim)
    def all(self, dim=None): from . import ops; return ops.all(self, dim=dim)
    def any(self, dim=None): from . import ops; return ops.any(self, dim=dim)
    def norm(self, p=2, dim=None, keepdim=False): from . import ops; return ops.norm(self, p=p, dim=dim, keepdim=keepdim)
    def topk(self, k, dim=-1, largest=True, sorted=True): from . import ops; return ops.topk(self, k, dim=dim, largest=largest, sorted=sorted)
    def sort(self, dim=-1, descending=False, stable=False): from . import ops; return ops.sort(self, dim=dim, descending=descending, stable=stable)

    # --- Indexing delegation ---
    def flip(self, dims): from . import ops; return ops.flip(self, dims)
    def roll(self, shifts, dims=None): from . import ops; return ops.roll(self, shifts, dims)
    def gather(self, dim, index): from . import ops; return ops.gather(self, dim, index)
    def chunk(self, chunks, dim=0): from . import ops; return ops.chunk(self, chunks, dim)
    def split(self, split_size, dim=0): from . import ops; return ops.split(self, split_size, dim)
    def nonzero(self, as_tuple=False): from . import ops; return ops.nonzero(self, as_tuple=as_tuple)
    def tril(self, diagonal=0): from . import ops; return ops.tril(self, diagonal)
    def triu(self, diagonal=0): from . import ops; return ops.triu(self, diagonal)

    # --- Comparison method aliases ---
    def eq(self, other): return self.__eq__(other)
    def ne(self, other): return self.__ne__(other)
    def lt(self, other): return self.__lt__(other)
    def le(self, other): return self.__le__(other)
    def gt(self, other): return self.__gt__(other)
    def ge(self, other): return self.__ge__(other)
    def isnan(self): from . import ops; return ops.isnan(self)
    def isinf(self): from . import ops; return ops.isinf(self)
    def isfinite(self): from . import ops; return ops.isfinite(self)
