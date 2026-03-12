from .tensor import Tensor
from . import ops
from .ops import add, sub, mul, div, matmul, sum, mean, relu, mm, cat, stack, log, exp, pow
import candle

def tensor(data, dtype=None, device=None, requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def ones(shape, dtype=None, device=None, requires_grad=False):
    t = candle.ones(shape)
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros(shape, dtype=None, device=None, requires_grad=False):
    t = candle.zeros(shape)
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

def randn(shape, dtype=None, device=None, requires_grad=False):
    t = candle.randn(shape)
    return Tensor(t, dtype=dtype, device=device, requires_grad=requires_grad)

# Export dtypes
float32 = candle.f32
float64 = candle.f64
int64 = candle.i64
# Add more as needed
