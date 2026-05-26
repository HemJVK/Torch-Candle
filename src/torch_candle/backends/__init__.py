# torch_candle hardware acceleration backends registry

from .registry import (
    onednn,
    mkl,
    mps,
    cuda,
)
