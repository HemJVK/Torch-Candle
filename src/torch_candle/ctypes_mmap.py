import ctypes
import mmap
import os

class TaskMetadata(ctypes.Structure):
    _fields_ = [
        ("op_code", ctypes.c_uint32),
        ("device_id", ctypes.c_uint32),
        ("input_size", ctypes.c_uint64),
        ("output_size", ctypes.c_uint64),
        ("payload", ctypes.c_ubyte * 256)
    ]

class CacheAlignedAtomicUsize(ctypes.Structure):
    _align_ = 128
    _fields_ = [
        ("val", ctypes.c_size_t)
    ]

class SPSCRingBufferLayout(ctypes.Structure):
    _fields_ = [
        ("head", CacheAlignedAtomicUsize),
        ("tail", CacheAlignedAtomicUsize),
        ("buffer", TaskMetadata * 1024)
    ]

class MmappedRingBuffer:
    def __init__(self, path="/dev/shm/torch_candle_ipc"):
        self.size = ctypes.sizeof(SPSCRingBufferLayout)
        if not os.path.exists(path):
            # Create file with correct size
            with open(path, "wb") as f:
                f.write(b"\x00" * self.size)
        self.fd = os.open(path, os.O_RDWR)
        self.mmap_obj = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.layout = SPSCRingBufferLayout.from_buffer(self.mmap_obj)

    def close(self):
        self.mmap_obj.close()
        os.close(self.fd)
