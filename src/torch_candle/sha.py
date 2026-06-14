import torch_candle_backend as _kernels
from .tensor import Tensor

def bypass(val: bool):
    """
    Disable (bypass=True) or Enable (bypass=False) the Self-Healing Autograd (SHA) engine.
    When bypassed, we also clear any stored gradient histories in memory.
    """
    enable_sha = not val
    Tensor.enable_sha = enable_sha
    _kernels.set_enable_sha(enable_sha)
    if val:
        _kernels.clear_grad_history()
