"""
torch_candle.Tensor — zero-copy thin wrapper around candle.Tensor (Rust/PyO3).
Hot paths are 100% candle Rust ops; numpy is only used where candle has no native equivalent.
"""
import numpy as np
import math as _math
import torch_candle_backend as _kernels

def _raw(t):
    """Return the underlying PyTensor from a Tensor wrapper."""
    return t._tensor if hasattr(t, '_tensor') else t

class Tensor:
    """torch_candle.Tensor — thin wrapper around candle.Tensor (Rust via PyO3).
    """

    __slots__ = ['_tensor', '_device', '_dtype', '_shape']

    _grad_enabled = True  # toggled by torch.no_grad()

    def __hash__(self):
        return id(self._tensor)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return self._tensor is other._tensor

    # ──────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────
    def __init__(self, data, dtype="float32", device="cpu", requires_grad=False):
        if device is None: device = "cpu"
        if dtype is None: dtype = "float32"
        if isinstance(data, Tensor):
            if requires_grad and not data.requires_grad:
                # Recreate leaf tensor with gradients
                self._tensor = _kernels.PyTensor(data.numpy(), device=data.device, dtype=data.dtype, requires_grad=True)
            else:
                self._tensor = data._tensor
        elif isinstance(data, _kernels.PyTensor):
            if requires_grad and not data.requires_grad:
                self._tensor = _kernels.PyTensor(data.to_numpy(), device=data.device, dtype="float32", requires_grad=True)
            else:
                self._tensor = data
        else:
            if isinstance(data, (list, tuple, np.ndarray, float, int, np.float32, np.float64)):
                arr = np.array(data, dtype=np.float32)
            else:
                arr = np.array(data).astype(np.float32)
            self._tensor = _kernels.PyTensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)

        # Cache properties from Rust core
        self._device = self._tensor.device
        self._dtype = "float32"
        self._shape = tuple(self._tensor.shape)

    @classmethod
    def _fast_wrap(cls, rust_tensor):
        """Internal fast construction bypassing __init__ overhead."""
        obj = cls.__new__(cls)
        obj._tensor = rust_tensor
        obj._device = rust_tensor.device
        obj._dtype = "float32"
        obj._shape = tuple(rust_tensor.shape)
        return obj

    # ──────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────
    @property
    def data(self):
        return self

    @property
    def shape(self):
        return self._shape

    def size(self, dim=None):
        return self._shape if dim is None else self._shape[dim]

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def requires_grad(self):
        return self._tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._tensor.requires_grad = bool(value)

    @property
    def grad(self):
        g = self._tensor.grad
        return self._fast_wrap(g) if g is not None else None

    @grad.setter
    def grad(self, value):
        if value is None:
            self._tensor.grad = None
        elif isinstance(value, Tensor):
            self._tensor.grad = value._tensor
        else:
            self._tensor.grad = Tensor(value, device=self.device)._tensor

    @property
    def grad_fn(self):
        return getattr(self._tensor, "grad_fn", None)

    # ──────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────
    def __getitem__(self, index):
        # Basic support for integer indexing to satisfy autograd tests
        if isinstance(index, int):
            if index < 0:
                index = self.shape[0] + index
            from . import ops
            idx_t = Tensor([index], dtype="uint32", device=self.device)
            return ops.index_select(self, 0, idx_t).squeeze(0)
        
        if isinstance(index, tuple):
            curr = self
            # Simplistic handling for tuple of ints like [0, 0]
            if all(isinstance(i, int) for i in index):
                from . import ops
                for dim, i in enumerate(index):
                    if i < 0:
                        i = curr.shape[0] + i
                    idx_t = Tensor([i], dtype="uint32", device=curr.device)
                    curr = ops.index_select(curr, 0, idx_t).squeeze(0)
                return curr

        # Fallback to numpy for complex slicing, no autograd support for these yet.
        out = Tensor(self.numpy()[index], device=self.device, dtype=self.dtype)
        return out

    def __setitem__(self, index, value):
        req_grad = self.requires_grad
        np_data = self.numpy()
        val_data = value.numpy() if isinstance(value, Tensor) else value
        np_data[index] = val_data
        self._tensor = _kernels.PyTensor(np_data.astype(np.float32), device=self.device, dtype=self.dtype, requires_grad=req_grad)

    def __repr__(self):
        return f"torch_candle.Tensor(shape={list(self.shape)}, dtype={self.dtype}, requires_grad={self.requires_grad})"

    # ──────────────────────────────────────────────────────────────
    # Arithmetic — delegated to Rust Autograd
    # ──────────────────────────────────────────────────────────────
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self._fast_wrap(self._tensor.add(other._tensor))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self._fast_wrap(self._tensor.sub(other._tensor))

    def __rsub__(self, other):
        return Tensor(other, device=self.device) - self

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self._fast_wrap(self._tensor.mul(other._tensor))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self._fast_wrap(self._tensor.div(other._tensor))

    def __rtruediv__(self, other):
        return Tensor(other, device=self.device) / self

    def __pow__(self, exponent):
        if isinstance(exponent, (int, float)):
            return self._fast_wrap(self._tensor.pow(float(exponent)))
        # For tensor exponent, we'd need a more complex native implementation
        return Tensor(self.numpy() ** (exponent.numpy() if isinstance(exponent, Tensor) else exponent), 
                      device=self.device, dtype=self.dtype)

    # ──────────────────────────────────────────────────────────────
    # Linear algebra
    # ──────────────────────────────────────────────────────────────
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        return self.matmul(other)

    def __rmatmul__(self, other):
        return Tensor(other, device=self.device).matmul(self)

    def matmul(self, other):
        return self._fast_wrap(self._tensor.matmul(_raw(other)))

    def t(self):
        return self._fast_wrap(self._tensor.t())

    def transpose(self, dim0, dim1):
        return self._fast_wrap(self._tensor.transpose(dim0, dim1))

    # ──────────────────────────────────────────────────────────────
    # Shape manipulation
    # ──────────────────────────────────────────────────────────────
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return self._fast_wrap(self._tensor.reshape(shape))

    def reshape(self, *shape):
        return self.view(*shape)

    def squeeze(self, dim=None):
        if dim is None:
            res = self._tensor
            for i, s in reversed(list(enumerate(self.shape))):
                if s == 1:
                    res = res.squeeze(i)
            return self._fast_wrap(res)
        return self._fast_wrap(self._tensor.squeeze(dim))

    def unsqueeze(self, dim):
        return self._fast_wrap(self._tensor.unsqueeze(dim))

    def flatten(self, start_dim=0, end_dim=-1):
        if start_dim == 0 and end_dim == -1:
            return self._fast_wrap(self._tensor.flatten_all())
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

    # ──────────────────────────────────────────────────────────────
    # Reductions
    # ──────────────────────────────────────────────────────────────
    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return self._fast_wrap(self._tensor.sum_all())
        # NumPy fallback for now to avoid recursion with ops.sum
        if isinstance(dim, list): dim = tuple(dim)
        res = np.sum(self.numpy(), axis=dim, keepdims=keepdim)
        return Tensor(res, device=self.device, dtype=self.dtype)

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return self._fast_wrap(self._tensor.mean_all())
        # Use sum then divide to maintain some consistency
        s = self.sum(dim=dim, keepdim=keepdim)
        shape = self.shape
        if isinstance(dim, int):
            size = shape[dim]
        elif isinstance(dim, (list, tuple)):
            size = 1
            for d in dim: size *= shape[d]
        else:
            size = self.numel()
        return s * (1.0 / size)

    # ──────────────────────────────────────────────────────────────
    # Unary element-wise
    # ──────────────────────────────────────────────────────────────
    def sqrt(self): from . import ops; return ops.sqrt(self)
    def exp(self):  from . import ops; return ops.exp(self)
    def log(self):  from . import ops; return ops.log(self)
    def relu(self): return self._fast_wrap(self._tensor.relu())
    def sin(self):  return self._fast_wrap(self._tensor.sin())
    def cos(self):  return self._fast_wrap(self._tensor.cos())
    def reciprocal(self): return self._fast_wrap(self._tensor.recip())

    def sigmoid(self):
        from . import ops
        return ops.sigmoid(self)

    def tanh(self):
        from . import ops
        return ops.tanh(self)

    # ──────────────────────────────────────────────────────────────
    # Device / dtype / Autograd
    # ──────────────────────────────────────────────────────────────
    def to(self, *args, **kwargs):
        device = self.device
        dtype = self.dtype
        for arg in args:
            if isinstance(arg, str):
                if arg in ("cpu", "cuda", "metal"): device = arg
                else: dtype = arg
        if device != self.device or dtype != self.dtype:
            new_t = Tensor(self.numpy(), device=device, dtype=dtype, requires_grad=self.requires_grad)
            return new_t
        return self

    def backward(self, gradient=None):
        grad_tensor = None
        if gradient is not None:
            if isinstance(gradient, Tensor): grad_tensor = gradient._tensor
            else: grad_tensor = Tensor(gradient, device=self.device)._tensor
        self._tensor.backward(grad_tensor)

    def detach(self):
        return self._fast_wrap(self._tensor.detach())

    def detach_(self):
        self._tensor.requires_grad = False
        return self

    def zero_(self):
        self._tensor = _kernels.PyTensor.zeros(self.shape, device=self.device, dtype=self.dtype)
        return self

    def numpy(self):
        return self._tensor.to_numpy()

    def item(self):
        n = self.numpy()
        return n.item() if hasattr(n, 'item') else float(n)

    def clone(self):
        return self._fast_wrap(self._tensor.clone())

    # ──────────────────────────────────────────────────────────────
    # Comparison
    # ──────────────────────────────────────────────────────────────
    def _cmp_np(self, op, other):
        fn = getattr(np, op)
        rhs = other.numpy() if isinstance(other, Tensor) else other
        return Tensor(fn(self.numpy(), rhs).astype(np.float32))

    def __eq__(self, other): return self._cmp_np('equal', other)
    def __ne__(self, other): return self._cmp_np('not_equal', other)
    def __lt__(self, other): return self._cmp_np('less', other)
    def __le__(self, other): return self._cmp_np('less_equal', other)
    def __gt__(self, other): return self._cmp_np('greater', other)
    def __ge__(self, other): return self._cmp_np('greater_equal', other)

    # ──────────────────────────────────────────────────────────────
    # Delegated trig / math / reduction / indexing
    # ──────────────────────────────────────────────────────────────
    def tan(self): return self.sin() / self.cos()
    def floor(self): from . import ops; return ops.floor(self)
    def ceil(self): from . import ops; return ops.ceil(self)
    
    def max(self, dim=None, keepdim=False): from . import ops; return ops.max(self, dim, keepdim)
    def min(self, dim=None, keepdim=False): from . import ops; return ops.min(self, dim, keepdim)
    
    def __len__(self): return self.shape[0] if len(self.shape) > 0 else 1
    def __iter__(self):
        for i in range(len(self)): yield self[i]

    def numel(self):
        return int(np.prod(self.shape))

    def abs(self):
        from . import ops
        return ops.abs(self)

    def clamp(self, min, max):
        from . import ops
        return ops.clamp(self, min, max)

    def std(self, dim=None, keepdim=False, unbiased=True):
        # Use numpy for std for now
        res = np.std(self.numpy(), axis=dim, keepdims=keepdim, ddof=1 if unbiased else 0)
        return Tensor(res, device=self.device, dtype=self.dtype)
