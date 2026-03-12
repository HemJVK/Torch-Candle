from .tensor import Tensor
import candle

def add(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    input_tensor = input._tensor if isinstance(input, Tensor) else input
    other_tensor = other._tensor if isinstance(other, Tensor) else other
    return Tensor(input_tensor + other_tensor)

def sub(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    input_tensor = input._tensor if isinstance(input, Tensor) else input
    other_tensor = other._tensor if isinstance(other, Tensor) else other
    return Tensor(input_tensor - other_tensor)

def mul(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    input_tensor = input._tensor if isinstance(input, Tensor) else input
    other_tensor = other._tensor if isinstance(other, Tensor) else other
    return Tensor(input_tensor * other_tensor)

def div(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    input_tensor = input._tensor if isinstance(input, Tensor) else input
    other_tensor = other._tensor if isinstance(other, Tensor) else other
    return Tensor(input_tensor / other_tensor)

def matmul(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    return Tensor(input._tensor.matmul(other._tensor))

def sum(input, dim=None, keepdim=False):
    if dim is None:
        return Tensor(input._tensor.sum_all())
    else:
        # Candle sum_keepdim or sum
        if keepdim:
            return Tensor(input._tensor.sum_keepdim(dim))
        else:
            t = input._tensor.sum_keepdim(dim)
            return Tensor(t.squeeze(dim))

def mean(input, dim=None, keepdim=False):
    if dim is None:
        return Tensor(input._tensor.mean_all())
    else:
        # candle might not have mean_keepdim directly, compute as sum / size
        s = sum(input, dim=dim, keepdim=keepdim)
        size = input.shape[dim]
        return s / size

def relu(input):
    return Tensor(input._tensor.broadcast_maximum(candle.Tensor(0.0).to_device(input.device).to_dtype(input.dtype)))

def mm(input, other):
    return matmul(input, other)

def cat(tensors, dim=0):
    candle_tensors = [t._tensor for t in tensors]
    return Tensor(candle.cat(candle_tensors, dim))

def stack(tensors, dim=0):
    candle_tensors = [t._tensor for t in tensors]
    return Tensor(candle.stack(candle_tensors, dim))

def log(input):
    return Tensor(input._tensor.log())

def exp(input):
    return Tensor(input._tensor.exp())

def pow(input, exponent):
    return Tensor(input._tensor.powf(exponent))
