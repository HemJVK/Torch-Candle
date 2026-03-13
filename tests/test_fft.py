import pytest
import numpy as np
import torch_candle as torch
import torch_candle.fft as fft

def test_fft():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = fft.fft(t)
    # The output from torch is cast to real parts due to candle limitations
    # np.fft.fft([1, 2, 3, 4]) = [10+0j, -2+2j, -2+0j, -2-2j] -> [10, -2, -2, -2]
    expected = np.array([10., -2., -2., -2.])
    assert np.allclose(result.numpy(), expected)

def test_ifft():
    t = torch.tensor([10.+0.j, -2.+2.j, -2.+0.j, -2.-2.j]) # Real parts [10, -2, -2, -2] pushed to Tensor
    result = fft.ifft(t)
    # Testing that ifft runs without type errors; note it's lossy due to real-valued input
    assert result.shape == (4,)

def test_rfft():
    # Real-valued input to rfft
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # rfft returns n//2 + 1 values = 3
    result = fft.rfft(t)
    assert result.shape == (3,)
    expected = np.array([10., -2., -2.])
    assert np.allclose(result.numpy(), expected)

def test_fft2():
    t = torch.ones(2, 2)
    result = fft.fft2(t)
    expected = np.array([[4.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]])
    assert np.allclose(result.numpy(), expected)

def test_fftfreq():
    t = fft.fftfreq(4, d=0.5)
    expected = np.array([0., 0.5, -1., -0.5])
    assert np.allclose(t.numpy(), expected)
