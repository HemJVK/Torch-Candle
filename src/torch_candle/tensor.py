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
        return self._tensor.shape

    @property
    def device(self):
        return self._tensor.device

    @property
    def dtype(self):
        return self._tensor.dtype

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
        # 2D transpose
        return Tensor(self._tensor.t())

    def transpose(self, dim0, dim1):
        return Tensor(self._tensor.transpose(dim0, dim1))

    def sum(self, dim=None, keepdim=False):
        return ops.sum(self, dim=dim, keepdim=keepdim)

    def mean(self, dim=None, keepdim=False):
        return ops.mean(self, dim=dim, keepdim=keepdim)

    def relu(self):
        return ops.relu(self)

    def exp(self):
        return Tensor(self._tensor.exp())

    def backward(self):
        # Placeholder for autograd integration
        # Candle has its own autograd if used via functional
        pass

    def numpy(self):
        return np.array(self._tensor.to_torch().numpy())

    def item(self):
        return self.numpy().item()
