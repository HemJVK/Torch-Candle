"""torch_candle.amp — Automatic Mixed Precision (AMP) matching torch.amp."""
import torch_candle as torch

class autocast:
    """Context manager for automatic mixed precision (AMP) in torch-candle."""
    def __init__(self, device_type='cuda', dtype='float16', enabled=True, cache_enabled=True):
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        self.cache_enabled = cache_enabled
        self.prev_enabled = False
        self.prev_dtype = "float16"

    def __enter__(self):
        if self.enabled:
            # We access the class-level properties of Tensor dynamically to avoid circular imports
            self.prev_enabled = torch.Tensor._amp_enabled
            self.prev_dtype = torch.Tensor._amp_dtype
            torch.Tensor._amp_enabled = True
            torch.Tensor._amp_dtype = self.dtype
        return self

    def __exit__(self, *args):
        if self.enabled:
            torch.Tensor._amp_enabled = self.prev_enabled
            torch.Tensor._amp_dtype = self.prev_dtype

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorated


class GradScaler:
    """Gradient scaler for mixed precision to prevent gradient underflow."""
    def __init__(self, device='cuda', init_scale=65536.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._found_inf = False

    def scale(self, outputs):
        if not self._enabled:
            return outputs
        return outputs * self._scale

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        pass

    def get_scale(self): return self._scale
    def get_growth_factor(self): return self._growth_factor
    def get_backoff_factor(self): return self._backoff_factor
    def get_growth_interval(self): return self._growth_interval
    def is_enabled(self): return self._enabled
    def state_dict(self): return {'scale': self._scale}
    def load_state_dict(self, d): self._scale = d.get('scale', self._scale)


# torch.cuda.amp compat
cuda = type('cuda_amp', (), {'autocast': autocast, 'GradScaler': GradScaler})()
