import candle
import numpy as np

class Tensor:
    def __init__(self, data, dtype=None, device=None, requires_grad=False):
        if isinstance(data, candle.Tensor):
            self._tensor = data
        elif isinstance(data, (list, tuple, np.ndarray, float, int)):
            # Convert to candle tensor
            # Note: candle handles numpy conversion well
            self._tensor = candle.Tensor(data)
        elif isinstance(data, Tensor):
            self._tensor = data._tensor
        else:
            raise TypeError(f"Cannot construct Tensor from {type(data)}")

        if dtype is not None:
            self._tensor = self._tensor.to_dtype(dtype)
        if device is not None:
            self._tensor = self._tensor.to_device(device)
            
        self.requires_grad = requires_grad
        self.grad = None

    @property
    def shape(self):
        # Return a torch-like shape (tuple)
        return self._tensor.shape

    @property
    def size(self):
        # size() is both a property and a method in PyTorch
        return self.shape

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

    def __repr__(self):
        return f"torch_candle.Tensor({self._tensor.__repr__()}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        return Tensor(self._tensor + other_tensor)

    def __sub__(self, other):
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        return Tensor(self._tensor - other_tensor)

    def __mul__(self, other):
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        return Tensor(self._tensor * other_tensor)

    def __truediv__(self, other):
        other_tensor = other._tensor if isinstance(other, Tensor) else other
        return Tensor(self._tensor / other_tensor)

    def matmul(self, other):
        return Tensor(self._tensor.matmul(other._tensor))

    def t(self):
        return Tensor(self._tensor.t())

    def transpose(self, dim0, dim1):
        return Tensor(self._tensor.transpose(dim0, dim1))

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self._tensor.sum_all())
        if keepdim:
            return Tensor(self._tensor.sum_keepdim(dim))
        t = self._tensor.sum_keepdim(dim)
        return Tensor(t.squeeze(dim))

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self._tensor.mean_all())
        s = self.sum(dim=dim, keepdim=keepdim)
        size = self.shape[dim]
        return s / size

    def relu(self):
        import candle
        # Use broadcast_maximum or functional if available
        return Tensor(self._tensor.broadcast_maximum(candle.Tensor(0.0).to_device(self.device).to_dtype(self.dtype)))

    def exp(self):
        return Tensor(self._tensor.exp())

    def log(self):
        return Tensor(self._tensor.log())

    def view(self, *shape):
        return Tensor(self._tensor.reshape(*shape))

    def reshape(self, *shape):
        return Tensor(self._tensor.reshape(*shape))

    def squeeze(self, dim=None):
        if dim is None:
            # candle squeeze might require dim
            # If so, we'd need to loop over dims of size 1
            res = self._tensor
            for i, s in enumerate(self.shape):
                if s == 1:
                    res = res.squeeze(i)
            return Tensor(res)
        return Tensor(self._tensor.squeeze(dim))

    def unsqueeze(self, dim):
        return Tensor(self._tensor.unsqueeze(dim))

    def to(self, *args, **kwargs):
        # args can be device, dtype, or another tensor
        # For simplicity, handle basic cases
        t = self._tensor
        for arg in args:
            if isinstance(arg, (str, candle.Device)):
                t = t.to_device(arg)
            elif isinstance(arg, candle.DType):
                t = t.to_dtype(arg)
            elif hasattr(arg, '_candle_device'): # Our device class
                t = t.to_device(arg._candle_device)
        return Tensor(t, requires_grad=self.requires_grad)

    def backward(self):
        # autograd integration
        pass

    def numpy(self):
        return np.array(self._tensor.to_torch().numpy())

    def item(self):
        return self.numpy().item()
