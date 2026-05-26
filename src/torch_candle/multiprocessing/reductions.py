from multiprocessing.shared_memory import SharedMemory
import numpy as np
from torch_candle import Tensor

def reduce_tensor(t):
    """Serialize tensor metadata and the shared memory segment handle."""
    if not t.is_shared():
        t.share_memory_()
    return (reconstruct_tensor, (t._shm.name, t.shape, t.dtype, t.requires_grad))

def reconstruct_tensor(shm_name, shape, dtype, requires_grad):
    """Attach to the shared memory segment in the receiving process and wrap in a zero-copy Tensor."""
    shm = SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    
    t = Tensor(arr, dtype=dtype, requires_grad=requires_grad)
    t._shm = shm
    return t
