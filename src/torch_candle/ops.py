from .tensor import Tensor
import candle

def add(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    if not isinstance(input, Tensor):
        input = Tensor(input)
    return input + other

def sub(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    if not isinstance(input, Tensor):
        input = Tensor(input)
    return input - other

def mul(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    if not isinstance(input, Tensor):
        input = Tensor(input)
    return input * other

def div(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    if not isinstance(input, Tensor):
        input = Tensor(input)
    return input / other

def matmul(input, other, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    return input.matmul(other)

def mm(input, other):
    return input.matmul(other)

def sum(input, dim=None, keepdim=False):
    return input.sum(dim=dim, keepdim=keepdim)

def mean(input, dim=None, keepdim=False):
    return input.mean(dim=dim, keepdim=keepdim)

def relu(input):
    return input.relu()

def cat(tensors, dim=0, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    candle_tensors = [t._tensor for t in tensors]
    return Tensor(candle.cat(candle_tensors, dim))

def stack(tensors, dim=0, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    candle_tensors = [t._tensor for t in tensors]
    return Tensor(candle.stack(candle_tensors, dim))

def log(input, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    return input.log()

def exp(input, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    return input.exp()

def pow(input, exponent, out=None):
    if out is not None:
        raise NotImplementedError("out= parameter not supported yet")
    return Tensor(input._tensor.powf(exponent))

def view(input, *shape):
    return input.view(*shape)

def reshape(input, *shape):
    return input.reshape(*shape)

def squeeze(input, dim=None):
    return input.squeeze(dim)

def unsqueeze(input, dim):
    return input.unsqueeze(dim)
