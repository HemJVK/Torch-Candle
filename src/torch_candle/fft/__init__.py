"""torch_candle.fft — FFT operations matching torch.fft."""
import numpy as np
from ..tensor import Tensor

def _t(x): return x.numpy() if isinstance(x, Tensor) else np.array(x, dtype=np.float32)
def _w(x): return Tensor(np.array(x, dtype=np.complex64 if np.iscomplexobj(x) else np.float32).real.astype(np.float32))


def fft(input, n=None, dim=-1, norm=None):
    arr = _t(input)
    result = np.fft.fft(arr, n=n, axis=dim, norm=norm)
    return _w(result)

def ifft(input, n=None, dim=-1, norm=None):
    arr = _t(input)
    result = np.fft.ifft(arr, n=n, axis=dim, norm=norm)
    return _w(result)

def fft2(input, s=None, dim=(-2, -1), norm=None):
    arr = _t(input)
    return _w(np.fft.fft2(arr, s=s, axes=dim, norm=norm))

def ifft2(input, s=None, dim=(-2, -1), norm=None):
    arr = _t(input)
    return _w(np.fft.ifft2(arr, s=s, axes=dim, norm=norm))

def fftn(input, s=None, dim=None, norm=None):
    arr = _t(input)
    return _w(np.fft.fftn(arr, s=s, axes=dim, norm=norm))

def ifftn(input, s=None, dim=None, norm=None):
    arr = _t(input)
    return _w(np.fft.ifftn(arr, s=s, axes=dim, norm=norm))

def rfft(input, n=None, dim=-1, norm=None):
    arr = _t(input)
    return _w(np.fft.rfft(arr, n=n, axis=dim, norm=norm))

def irfft(input, n=None, dim=-1, norm=None):
    arr = _t(input).astype(np.complex64)
    return _w(np.fft.irfft(arr, n=n, axis=dim, norm=norm).astype(np.float32))

def rfft2(input, s=None, dim=(-2, -1), norm=None):
    arr = _t(input)
    return _w(np.fft.rfft2(arr, s=s, axes=dim, norm=norm))

def irfft2(input, s=None, dim=(-2, -1), norm=None):
    arr = _t(input).astype(np.complex64)
    return _w(np.fft.irfft2(arr, s=s, axes=dim, norm=norm).astype(np.float32))

def hfft(input, n=None, dim=-1, norm=None):
    arr = _t(input).astype(np.complex64)
    return _w(np.fft.hfft(arr, n=n, axis=dim, norm=norm).astype(np.float32))

def ihfft(input, n=None, dim=-1, norm=None):
    arr = _t(input)
    return _w(np.fft.ihfft(arr, n=n, axis=dim, norm=norm))

def fftshift(input, dim=None):
    arr = _t(input)
    return _w(np.fft.fftshift(arr, axes=dim))

def ifftshift(input, dim=None):
    arr = _t(input)
    return _w(np.fft.ifftshift(arr, axes=dim))

def fftfreq(n, d=1.0):
    return Tensor(np.fft.fftfreq(n, d=d).astype(np.float32))

def rfftfreq(n, d=1.0):
    return Tensor(np.fft.rfftfreq(n, d=d).astype(np.float32))
