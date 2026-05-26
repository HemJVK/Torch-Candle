class Device:
    """c10 device type representation."""
    def __init__(self, device_str):
        self.device_str = device_str
        self.type = device_str.split(":")[0]
        self.index = int(device_str.split(":")[1]) if ":" in device_str else 0
        
    def __repr__(self):
        return f"c10.Device({self.type}:{self.index})"

class DispatchKey:
    """c10 hardware dispatch target keys."""
    CPU = "CPU"
    CUDA = "CUDA"
    ROCM = "ROCM"
    XPU = "XPU"
    MPS = "MPS"

class Allocator:
    """c10 tensor memory allocation registry and allocator types."""
    def __init__(self, name="default"):
        self.name = name
        self.allocated_blocks = 0
        
    def allocate(self, size):
        self.allocated_blocks += 1
        return f"ptr_to_{size}_bytes"

_current_allocator = Allocator("default")

def set_allocator(allocator):
    global _current_allocator
    _current_allocator = allocator

def get_allocator():
    global _current_allocator
    return _current_allocator
