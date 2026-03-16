"""torch_candle.cuda — CUDA utility functions matching torch.cuda API."""

import torch_candle_backend as _kernels


def is_available():
    """Returns True if GPU/CUDA is available via Candle backend."""
    try:
        _kernels.PyTensor.ones([1], device="cuda", dtype="float32")
        return True
    except Exception:
        return False


def device_count():
    """Returns the number of CUDA devices available."""
    return 1 if is_available() else 0


def get_device_name(device=None):
    """Returns the name of the GPU device."""
    if is_available():
        return "NVIDIA GPU (via Candle)"
    return "CPU"


def current_device():
    """Returns the index of the current CUDA device (always 0)."""
    return 0


def set_device(device):
    """Sets the current CUDA device (no-op, Candle manages this)."""
    pass


def empty_cache():
    """Releases unused cached GPU memory (no-op stub)."""
    pass


def memory_allocated(device=None):
    """Returns the amount of GPU memory allocated (returns 0 — Candle manages memory)."""
    return 0


def memory_reserved(device=None):
    """Returns the amount of GPU memory reserved (returns 0)."""
    return 0


def max_memory_allocated(device=None):
    """Returns the maximum GPU memory allocated (returns 0)."""
    return 0


def reset_peak_memory_stats(device=None):
    """Resets peak memory stats (no-op)."""
    pass


def synchronize(device=None):
    """Synchronizes CUDA streams (no-op, Candle is synchronous)."""
    pass


def is_initialized():
    """Returns True if CUDA has been initialized."""
    return is_available()


def get_arch_list():
    """Returns list of CUDA architectures (stub)."""
    return []
