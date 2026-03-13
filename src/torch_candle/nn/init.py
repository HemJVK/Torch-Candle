"""
torch.nn.init — Parameter initialization routines.
All ops use numpy for computation and wrap in Tensor, mirroring PyTorch's nn.init API.
"""
import numpy as np
import math

def _check_tensor(tensor):
    from ..tensor import Tensor
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Expected torch_candle.Tensor, got {type(tensor)}")
    return tensor

# ============================================================
# Uniform / Normal
# ============================================================
def uniform_(tensor, a=0.0, b=1.0):
    """Fill tensor with values from U(a, b) in-place."""
    _check_tensor(tensor).uniform_(a, b); return tensor

def normal_(tensor, mean=0.0, std=1.0):
    """Fill tensor with values from N(mean, std) in-place."""
    _check_tensor(tensor).normal_(mean, std); return tensor

def constant_(tensor, val):
    """Fill tensor with val in-place."""
    _check_tensor(tensor).fill_(val); return tensor

def ones_(tensor):
    return constant_(tensor, 1.0)

def zeros_(tensor):
    return constant_(tensor, 0.0)

def eye_(tensor):
    t = _check_tensor(tensor)
    n, m = t.shape[0], t.shape[1] if len(t.shape) > 1 else t.shape[0]
    arr = np.eye(n, m, dtype=np.float32)
    from ..tensor import Tensor
    t._tensor = Tensor(arr.tolist(), device=t.device, dtype=t.dtype)._tensor; return tensor

def dirac_(tensor, groups=1):
    raise NotImplementedError("dirac_ not yet implemented")

# ============================================================
# Xavier (Glorot)
# ============================================================
def _calculate_fan_in_out(tensor):
    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError("Fan calculation requires tensor of at least 2 dims")
    fan_in = shape[1]
    fan_out = shape[0]
    if len(shape) > 2:
        receptive_field = 1
        for s in shape[2:]:
            receptive_field *= s
        fan_in *= receptive_field
        fan_out *= receptive_field
    return fan_in, fan_out

def _calculate_gain(nonlinearity, param=None):
    gains = {'sigmoid': 1, 'tanh': 5.0/3, 'relu': math.sqrt(2.0), 'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01)**2))}
    return gains.get(nonlinearity, 1.0)

def xavier_uniform_(tensor, gain=1.0):
    t = _check_tensor(tensor)
    fan_in, fan_out = _calculate_fan_in_out(t)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)

def xavier_normal_(tensor, gain=1.0):
    t = _check_tensor(tensor)
    fan_in, fan_out = _calculate_fan_in_out(t)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)

# ============================================================
# Kaiming (He)
# ============================================================
def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    t = _check_tensor(tensor)
    fan_in, fan_out = _calculate_fan_in_out(t)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    t = _check_tensor(tensor)
    fan_in, fan_out = _calculate_fan_in_out(t)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)

# ============================================================
# Orthogonal / Sparse
# ============================================================
def orthogonal_(tensor, gain=1.0):
    t = _check_tensor(tensor)
    rows, cols = t.shape[0], int(np.prod(t.shape[1:]))
    flattened = np.random.normal(0, 1, (rows, cols))
    if rows < cols:
        flattened = flattened.T
    q, r = np.linalg.qr(flattened)
    d = np.diag(r)
    q *= np.sign(d)
    if rows < cols:
        q = q.T
    q = (q * gain).astype(np.float32)
    from ..tensor import Tensor
    t._tensor = Tensor(q.reshape(t.shape).tolist(), device=t.device, dtype=t.dtype)._tensor
    return tensor

def sparse_(tensor, sparsity, std=0.01):
    t = _check_tensor(tensor)
    rows = t.shape[0]
    arr = np.zeros(t.shape, dtype=np.float32)
    for col in range(int(np.prod(t.shape[1:]))):
        num_zeros = int(rows * sparsity)
        zero_rows = np.random.choice(rows, num_zeros, replace=False)
        col_data = np.random.normal(0, std, rows).astype(np.float32)
        col_data[zero_rows] = 0.0
        arr[:, col] = col_data
    from ..tensor import Tensor
    t._tensor = Tensor(arr.tolist(), device=t.device, dtype=t.dtype)._tensor
    return tensor
