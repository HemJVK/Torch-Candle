from .module import Module
import candle

class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        if self.training:
            # candle might have functional dropout, if not we implement using mask
            # For now, let's use a simple mask
            import random
            mask_data = [1.0 if random.random() > self.p else 0.0 for _ in range(input.nelement)]
            mask = candle.Tensor(mask_data).reshape(input.shape).to_device(input.device)
            return input * mask * (1.0 / (1.0 - self.p))
        return input

    def __repr__(self):
        return f"Dropout(p={self.p}, inplace={self.inplace})"
