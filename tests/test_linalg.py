import pytest
import numpy as np
import torch_candle as torch
import torch_candle.linalg as linalg

def test_det_inv():
    # 2x2 matrix with determinant 2 = (4*1 - 2*1)
    t = torch.tensor([[4.0, 2.0], [1.0, 1.0]])
    det = linalg.det(t)
    assert abs(det.item() - 2.0) < 1e-5

    inv = linalg.inv(t)
    # expected inv = [[0.5, -1.0], [-0.5, 2.0]]
    expected = np.array([[0.5, -1.0], [-0.5, 2.0]])
    assert np.allclose(inv.numpy(), expected)

def test_qr():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    q, r = linalg.qr(t)
    # Test q * r = t
    res = q @ r
    assert np.allclose(res.numpy(), t.numpy(), atol=1e-5)

def test_svd():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    u, s, vh = linalg.svd(t)
    # reconst = u * diag(s) * vh
    s_mat = torch.Tensor(np.diag(s.numpy()).astype(np.float32))
    reconst = u @ s_mat @ vh
    assert np.allclose(reconst.numpy(), t.numpy(), atol=1e-5)

def test_cholesky():
    # Positive definite matrix
    t = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    l = linalg.cholesky(t)
    # l * l^T = t
    res = l @ l.t()
    assert np.allclose(res.numpy(), t.numpy(), atol=1e-5)

def test_solve_lstsq():
    A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    b = torch.tensor([5.0, 4.0])
    x = linalg.solve(A, b)
    # expected x = [2.0, 1.0]
    assert np.allclose(x.numpy(), [2.0, 1.0], atol=1e-5)

    A_lstsq = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    b_lstsq = torch.tensor([2.0, 0.0, 2.0])
    res = linalg.lstsq(A_lstsq, b_lstsq)
    x_lstsq = res[0]
    assert np.allclose(x_lstsq.numpy(), [1.0, 1.0], atol=1e-5)
