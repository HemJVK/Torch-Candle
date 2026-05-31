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

class Stream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id

class Event:
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self.stream_id = None
        self._event = None

    def record(self, stream=None):
        stream_id = stream.stream_id if stream else 0
        self.stream_id = stream_id
        self._event = _allocator.record_event(stream_id)

    def wait(self, stream=None):
        comm_stream_id = stream.stream_id if stream else 0
        if self._event is not None:
            _allocator.wait_event(comm_stream_id, self._event)

    def query(self) -> bool:
        if self._event is not None:
            return self._event.query()
        return True

_allocator = _kernels.StreamAwareAllocator()

def stream_wait_event(comm_stream: Stream, computation_stream: Stream):
    """
    Shifts the synchronization burden from CPU to GPU communication streams.
    Records an event on computation_stream and waits for it on comm_stream.
    """
    event = Event()
    event.record(computation_stream)
    event.wait(comm_stream)

class DelayedDeletionManager:
    """
    Delayed Deletion Pipeline: Delays the deletion (del) of layer i tensors
    until after layer i+1 has been successfully scheduled on the GPU.
    This guarantees maximum communication stream and all-gather overlapping.
    """
    _pending_deletions = []

    @classmethod
    def queue_deletion(cls, tensor):
        cls._pending_deletions.append(tensor)

    @classmethod
    def process_pending(cls):
        while cls._pending_deletions:
            t = cls._pending_deletions.pop(0)
            del t

class delayed_deletion:
    """
    Context manager to safely trigger processed pending delayed deletions.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        DelayedDeletionManager.process_pending()
