import multiprocessing
from multiprocessing import *
from multiprocessing.reduction import ForkingPickler
from torch_candle import Tensor
from .reductions import reduce_tensor, reconstruct_tensor

# Expose standard multiprocessing members
globals().update({k: v for k, v in multiprocessing.__dict__.items() if not k.startswith("__")})

# Register our custom Tensor reducer with Python's ForkingPickler
ForkingPickler.register(Tensor, reduce_tensor)
