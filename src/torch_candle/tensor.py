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

    __slots__ = ['_tensor', '_device', '_dtype', '_shape', '_id', '_shm']

    _grad_enabled = True  # toggled by torch.no_grad()
    enable_sha = False
    _grad_history = {}

    def __hash__(self):
        return id(self._tensor)

    def __eq__(self, other):
        if not isinstance(other, Tensor):
            return False
        return self._tensor is other._tensor

    # ──────────────────────────────────────────────────────────────
    # Class-level Automatic Mixed Precision flags
    _amp_enabled = False
    _amp_dtype = "float16"

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
                arr = np.ascontiguousarray(data, dtype=np.float32)
            else:
                arr = np.ascontiguousarray(np.array(data), dtype=np.float32)
            self._tensor = _kernels.PyTensor(arr, device=device, dtype=dtype, requires_grad=requires_grad)

        # Cache properties from Rust core
        self._device = self._tensor.device
        self._dtype = dtype
        self._shape = tuple(self._tensor.shape)

        self._shm = None

        # Unique ID for graph compiler tracing
        if not hasattr(Tensor, "_id_counter"):
            Tensor._id_counter = 0
        Tensor._id_counter += 1
        self._id = Tensor._id_counter

    @classmethod
    def _fast_wrap(cls, rust_tensor, dtype="float32"):
        """Internal fast construction bypassing __init__ overhead."""
        obj = cls.__new__(cls)
        obj._tensor = rust_tensor
        obj._device = rust_tensor.device
        obj._dtype = dtype
        obj._shape = tuple(rust_tensor.shape)

        obj._shm = None

        # Unique ID for graph compiler tracing
        if not hasattr(Tensor, "_id_counter"):
            Tensor._id_counter = 0
        Tensor._id_counter += 1
        obj._id = Tensor._id_counter

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
        val = bool(value)
        self._tensor.requires_grad = val
        if val:
            import torch_candle_backend as _kernels
            self._tensor = _kernels.PyTensor(self.numpy(), device=self.device, dtype=self.dtype, requires_grad=True)

    @property
    def grad(self):
        g = self._tensor.retrieve_grad(self._id)
        if g is None:
            return None
        return self._fast_wrap(g, dtype=self.dtype)

    @grad.setter
    def grad(self, value):
        if value is None:
            self._tensor.grad = None
            return
            
        if not isinstance(value, Tensor):
            value = Tensor(value, device=self.device)
            
        self._tensor.grad = value._tensor

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
        if self.device != other.device:
            other = other.to(self.device)
        return self._fast_wrap(self._tensor.add(other._tensor), dtype=self.dtype)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        if self.device != other.device:
            other = other.to(self.device)
        return self._fast_wrap(self._tensor.sub(other._tensor), dtype=self.dtype)

    def __rsub__(self, other):
        return Tensor(other, device=self.device) - self

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        if self.device != other.device:
            other = other.to(self.device)
        return self._fast_wrap(self._tensor.mul(other._tensor), dtype=self.dtype)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        if self.device != other.device:
            other = other.to(self.device)
        return self._fast_wrap(self._tensor.div(other._tensor), dtype=self.dtype)

    def __rtruediv__(self, other):
        return Tensor(other, device=self.device) / self

    def __iadd__(self, other):
        return self.add_(other)

    def __isub__(self, other):
        return self.sub_(other)

    def __imul__(self, other):
        return self.mul_(other)

    def __itruediv__(self, other):
        return self.div_(other)

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
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        if self.device != other.device:
            other = other.to(self.device)
        return self._fast_wrap(self._tensor.matmul(_raw(other)), dtype=self.dtype)

    def t(self):
        return self._fast_wrap(self._tensor.t())

    def transpose(self, dim0, dim1):
        d0 = dim0 if dim0 >= 0 else self.ndim + dim0
        d1 = dim1 if dim1 >= 0 else self.ndim + dim1
        return self._fast_wrap(self._tensor.transpose(d0, d1))

    def contiguous(self):
        return self._fast_wrap(self._tensor.contiguous(), dtype=self.dtype)

    # ──────────────────────────────────────────────────────────────
    # Shape manipulation
    # ──────────────────────────────────────────────────────────────
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        
        # Resolve negative dimension (-1) if present
        if -1 in shape:
            total_elements = self.numel()
            neg_idx = shape.index(-1)
            other_elements = 1
            for i, dim in enumerate(shape):
                if i != neg_idx:
                    other_elements *= dim
            if other_elements == 0:
                shape[neg_idx] = 0
            else:
                shape[neg_idx] = total_elements // other_elements
                
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
        d = dim if dim >= 0 else self.ndim + dim
        return self._fast_wrap(self._tensor.squeeze(d))

    def unsqueeze(self, dim):
        d = dim if dim >= 0 else self.ndim + dim
        return self._fast_wrap(self._tensor.unsqueeze(d))

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
        if isinstance(dim, (list, tuple)):
            dims = [d if d >= 0 else self.ndim + d for d in dim]
            res = self
            for d in sorted(dims, reverse=True):
                res = res._fast_wrap(res._tensor.sum_dim(d, keepdim))
            return res
        d = int(dim)
        if d < 0:
            d = self.ndim + d
        return self._fast_wrap(self._tensor.sum_dim(d, keepdim))

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return self._fast_wrap(self._tensor.mean_all())
        if isinstance(dim, (list, tuple)):
            dims = [d if d >= 0 else self.ndim + d for d in dim]
            res = self
            for d in sorted(dims, reverse=True):
                res = res._fast_wrap(res._tensor.mean_dim(d, keepdim))
            return res
        d = int(dim)
        if d < 0:
            d = self.ndim + d
        return self._fast_wrap(self._tensor.mean_dim(d, keepdim))

    # ──────────────────────────────────────────────────────────────
    # Unary element-wise
    # ──────────────────────────────────────────────────────────────
    def sqrt(self): return self._fast_wrap(self._tensor.sqrt())
    def exp(self):  return self._fast_wrap(self._tensor.exp())
    def log(self):  return self._fast_wrap(self._tensor.log())
    def relu(self): return self._fast_wrap(self._tensor.relu())
    def sin(self):  return self._fast_wrap(self._tensor.sin())
    def cos(self):  return self._fast_wrap(self._tensor.cos())
    def reciprocal(self): return self._fast_wrap(self._tensor.recip())

    def sigmoid(self): return self._fast_wrap(self._tensor.sigmoid())
    def tanh(self):    return self._fast_wrap(self._tensor.tanh())

    def erf(self):     return self._fast_wrap(self._tensor.erf())
    def neg(self):     return -self

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

    def float(self):
        return self.to("float32")

    def double(self):
        return self.to("float64")

    def half(self):
        return self.to("float16")

    def cuda(self):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def add_(self, other):
        res = self + other
        self._tensor = res._tensor
        return self

    def sub_(self, other):
        res = self - other
        self._tensor = res._tensor
        return self

    def mul_(self, other):
        res = self * other
        self._tensor = res._tensor
        return self

    def div_(self, other):
        res = self / other
        self._tensor = res._tensor
        return self

    def relu_(self):
        res = self.relu()
        self._tensor = res._tensor
        return self

    def sigmoid_(self):
        res = self.sigmoid()
        self._tensor = res._tensor
        return self

    def tanh_(self):
        res = self.tanh()
        self._tensor = res._tensor
        return self

    def backward(self, gradient=None, retain_graph=False):
        grad_tensor = None
        if gradient is not None:
            if isinstance(gradient, Tensor): grad_tensor = gradient._tensor
            else: grad_tensor = Tensor(gradient, device=self.device)._tensor
        self._tensor.backward(grad_tensor)

        # Run custom Python autograd function backwards if recorded on the tape
        from .autograd import Function
        if Function._tape:
            tape_copy = list(Function._tape)
            Function._tape.clear()
            
            for cls, ctx, inputs, output in reversed(tape_copy):
                out_grad = output.grad
                if out_grad is None:
                    from . import ones_like
                    out_grad = ones_like(output)
                    
                grads = cls.backward(ctx, out_grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)
                    
                for inp, g in zip(inputs, grads):
                    if isinstance(inp, Tensor) and inp.requires_grad and g is not None:
                        if not inp._tensor.has_grad_fn:
                            if inp.grad is None:
                                inp.grad = g
                            else:
                                inp.grad = inp.grad + g
                            
                        # Dynamically propagate gradient further backward through the Rust/Python graph
                        if inp._tensor.has_grad_fn:
                            inp.backward(g)

    def record_stream(self, stream):
        """
        Record that the tensor is being used by the given stream.
        Re-enrolls active caching allocator metadata tracking alongside GPU-native Events.
        """
        from torch_candle.cuda import Event, Stream, _allocator
        comp_stream = Stream(0)
        comm_stream = stream if isinstance(stream, Stream) else Stream(stream)
        
        event = Event()
        event.record(comp_stream)
        event.wait(comm_stream)
        
        _allocator.record_stream(id(self), comm_stream.stream_id)

    def __del__(self):
        try:
            from torch_candle.cuda import _allocator
            _allocator.free(id(self), 0)
            _allocator.cuda_free(id(self))
        except Exception:
            pass

    def detach(self):
        return Tensor(self.numpy(), device=self.device, dtype=self.dtype, requires_grad=False)

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

    def __getstate__(self):
        return {
            "data": self.numpy(),
            "device": self.device,
            "dtype": self.dtype,
            "requires_grad": self.requires_grad
        }

    def __setstate__(self, state):
        import torch_candle_backend as _kernels
        self._tensor = _kernels.PyTensor(
            state["data"],
            device=state["device"],
            dtype=state["dtype"],
            requires_grad=state["requires_grad"]
        )
        self._device = self._tensor.device
        self._dtype = "float32"
        self._shape = tuple(self._tensor.shape)
        self._shm = None

    def share_memory_(self):
        if self.is_shared():
            return self
        from multiprocessing.shared_memory import SharedMemory
        import numpy as np
        import torch_candle_backend as _kernels
        
        element_size = np.dtype(self.dtype).itemsize
        size = self.numel() * element_size
        
        shm = SharedMemory(create=True, size=size)
        self._shm = shm
        
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        arr[:] = self.numpy()[:]
        
        self._tensor = _kernels.PyTensor(arr, device=self.device, dtype=self.dtype, requires_grad=self.requires_grad)
        return self

    def is_shared(self):
        return self._shm is not None

    def __torch_dispatch__(self, func_name, *args, **kwargs):
        """
        Base subclass interception layer. Routes back to standard execution
        if no subclass overrides it.
        """
        return getattr(self, func_name)(*args, **kwargs)
