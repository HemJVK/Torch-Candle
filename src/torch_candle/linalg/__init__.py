"""torch_candle.linalg — maps to NumPy linalg (Candle Rust linalg when available)."""
import numpy as np
from ..tensor import Tensor

def _t(x): return x.numpy() if isinstance(x, Tensor) else x
def _w(x): return Tensor(np.array(x, dtype=np.float32))

def det(input): return _w(np.linalg.det(_t(input)))
def slogdet(input):
    sign, logabsdet = np.linalg.slogdet(_t(input))
    return _w(sign), _w(logabsdet)
def matrix_rank(input, tol=None, hermitian=False):
    return _w(float(np.linalg.matrix_rank(_t(input), tol=tol)))
def inv(input): return _w(np.linalg.inv(_t(input)).astype(np.float32))
def pinv(input, rcond=1e-15, hermitian=False):
    return _w(np.linalg.pinv(_t(input), rcond=rcond).astype(np.float32))
def solve(A, B): return _w(np.linalg.solve(_t(A), _t(B)).astype(np.float32))
def lstsq(A, B, rcond=None, driver=None):
    res, residuals, rank, sv = np.linalg.lstsq(_t(A), _t(B), rcond=rcond)
    return _w(res.astype(np.float32)), _w(residuals.astype(np.float32)), _w(float(rank)), _w(sv.astype(np.float32))

def svd(input, full_matrices=True, driver=None):
    U, S, Vh = np.linalg.svd(_t(input), full_matrices=full_matrices)
    return _w(U.astype(np.float32)), _w(S.astype(np.float32)), _w(Vh.astype(np.float32))

def eig(input):
    vals, vecs = np.linalg.eig(_t(input))
    return _w(vals.real.astype(np.float32)), _w(vecs.real.astype(np.float32))

def eigh(input, UPLO='L'):
    vals, vecs = np.linalg.eigh(_t(input), UPLO=UPLO)
    return _w(vals.astype(np.float32)), _w(vecs.astype(np.float32))

def eigvals(input): return _w(np.linalg.eigvals(_t(input)).real.astype(np.float32))
def eigvalsh(input, UPLO='L'): return _w(np.linalg.eigvalsh(_t(input), UPLO=UPLO).astype(np.float32))

def qr(input, mode='reduced'):
    Q, R = np.linalg.qr(_t(input), mode=mode)
    return _w(Q.astype(np.float32)), _w(R.astype(np.float32))

def cholesky(input): return _w(np.linalg.cholesky(_t(input)).astype(np.float32))

def norm(input, ord=None, dim=None, keepdim=False, dtype=None):
    return _w(np.linalg.norm(_t(input), ord=ord, axis=dim, keepdims=keepdim).astype(np.float32))

def vector_norm(input, ord=2, dim=None, keepdim=False):
    return _w(np.linalg.norm(_t(input), ord=ord, axis=dim, keepdims=keepdim).astype(np.float32))

def matrix_norm(input, ord='fro', dim=(-2, -1), keepdim=False):
    return _w(np.linalg.norm(_t(input), ord=ord, axis=dim, keepdims=keepdim).astype(np.float32))

def cross(input, other, dim=-1):
    return _w(np.cross(_t(input), _t(other), axis=dim).astype(np.float32))

def outer(input, vec2):
    return _w(np.outer(_t(input), _t(vec2)).astype(np.float32))
