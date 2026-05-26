import sys
from types import ModuleType

class MPSModule(ModuleType):
    def __init__(self):
        super().__init__("torch_candle.aten.mps")
        self.is_available = lambda: False
        self.is_built = lambda: False

mps = MPSModule()
sys.modules["torch_candle.aten.mps"] = mps
sys.modules["torch_candle.aten.mps"] = mps
