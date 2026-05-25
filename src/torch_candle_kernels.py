import torch_candle_backend as _backend
import numpy as np

def relu(x):
    """AVX2 vectorized parallel ReLU (in-place)."""
    _backend.fast_relu(x)

def sigmoid(x):
    """AVX2 vectorized parallel Sigmoid (in-place)."""
    _backend.fast_sigmoid(x)

def tanh(x):
    """AVX2 vectorized parallel Tanh (in-place)."""
    _backend.fast_tanh(x)

def silu(x):
    """AVX2 vectorized parallel SiLU (in-place)."""
    _backend.fast_silu(x)

def gelu(x):
    """AVX2 vectorized parallel GELU (in-place)."""
    _backend.fast_gelu(x)

def softmax(x, dim=-1):
    """AVX2 vectorized parallel Softmax (in-place)."""
    _backend.fast_softmax(x, dim)

def layer_norm(x, weight=None, bias=None, eps=1e-5):
    """AVX2 vectorized parallel LayerNorm (in-place)."""
    _backend.fast_layer_norm(x, weight, bias, eps)

def adam_step(param, grad, m, v, beta1, beta2, lr, eps, step):
    """AVX2 vectorized parallel Adam Optimizer Step (in-place)."""
    _backend.fast_adam_step(param, grad, m, v, beta1, beta2, lr, eps, step)
