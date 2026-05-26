import sys
from types import ModuleType

# oneDNN (MKLDNN) Backend Configuration
class OneDNNModule(ModuleType):
    def __init__(self):
        super().__init__("torch_candle.backends.onednn")
        self.enabled = True
        
    def is_available(self):
        return True
        
onednn = OneDNNModule()
sys.modules["torch_candle.backends.onednn"] = onednn

# MKL Backend Configuration
class MKLModule(ModuleType):
    def __init__(self):
        super().__init__("torch_candle.backends.mkl")
        
    def is_available(self):
        return True
        
mkl = MKLModule()
sys.modules["torch_candle.backends.mkl"] = mkl

# Apple MPS Backend Configuration
class MPSModule(ModuleType):
    def __init__(self):
        super().__init__("torch_candle.backends.mps")
        
    def is_available(self):
        return False
        
    def is_built(self):
        return False
        
mps = MPSModule()
sys.modules["torch_candle.backends.mps"] = mps

# NVIDIA CUDA Backend Configuration
class MatmulFlags:
    def __init__(self):
        self.allow_tf32 = True

class CUDAModule(ModuleType):
    def __init__(self):
        super().__init__("torch_candle.backends.cuda")
        self.matmul = MatmulFlags()
        
    def is_available(self):
        import torch_candle_backend as _kernels
        return hasattr(_kernels, "cuda_is_available") and _kernels.cuda_is_available()
        
cuda = CUDAModule()
sys.modules["torch_candle.backends.cuda"] = cuda
