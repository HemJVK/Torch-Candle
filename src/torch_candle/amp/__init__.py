"""torch_candle.amp — Automatic Mixed Precision (AMP) stubs matching torch.amp."""

class autocast:
    """Context manager for automatic mixed precision — no-op in torch_candle (Candle handles precision natively)."""
    def __init__(self, device_type='cpu', dtype=None, enabled=True, cache_enabled=True):
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled
        self.cache_enabled = cache_enabled

    def __enter__(self): return self

    def __exit__(self, *args): pass

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorated


class GradScaler:
    """Gradient scaler for mixed precision — no-op in torch_candle (Candle handles this natively)."""
    def __init__(self, device='cpu', init_scale=65536.0, growth_factor=2.0,
                 backoff_factor=0.5, growth_interval=2000, enabled=True):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._found_inf = False

    def scale(self, outputs):
        """Scale outputs (no-op)."""
        return outputs

    def unscale_(self, optimizer):
        """Unscale gradients (no-op)."""
        pass

    def step(self, optimizer, *args, **kwargs):
        """Optimizer step (delegates to optimizer.step)."""
        return optimizer.step(*args, **kwargs)

    def update(self, new_scale=None):
        """Update scale factor (no-op)."""
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
