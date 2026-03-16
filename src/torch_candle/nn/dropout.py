"""torch_candle.nn.dropout — Dropout layers."""
from .module import Module
import numpy as np




class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if self.training and self.p > 0:
            from ..tensor import Tensor
            x = input.numpy()
            mask = (np.random.random(x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
            return Tensor((x * mask).astype(np.float32))
        return input

    def __repr__(self):
        return f"Dropout(p={self.p}, inplace={self.inplace})"


class Dropout2d(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if self.training and self.p > 0:
            from ..tensor import Tensor
            x = input.numpy()
            # Channel-wise dropout for 4D tensors (N, C, H, W)
            mask_shape = x.shape[:2] + (1,) * (x.ndim - 2)
            mask = (np.random.random(mask_shape) > self.p).astype(np.float32) / (1.0 - self.p)
            return Tensor((x * mask).astype(np.float32))
        return input

    def __repr__(self):
        return f"Dropout2d(p={self.p}, inplace={self.inplace})"


class AlphaDropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if not self.training or self.p == 0:
            return input
        from ..tensor import Tensor
        alpha = -1.7580993408473766
        lam = 1.0507009873554805
        x = input.numpy()
        mask = np.random.random(x.shape) >= self.p
        a = (1 / (1 - self.p)) ** 0.5
        b = -a * lam * alpha * self.p
        out = np.where(mask, x, alpha * lam)
        return Tensor((a * out + b).astype(np.float32))
