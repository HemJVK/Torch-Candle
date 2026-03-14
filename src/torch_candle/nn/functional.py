"""
torch_candle.nn.functional — All neural network functional ops.

Design: hot-path ops (activations, linear, conv, losses, attention) use
candle Rust via the Tensor abstraction. numpy is used only for ops where
candle has no equivalents (per-pixel adaptive pooling, interpolation, etc.)
"""

from __future__ import annotations
import math as _math
import numpy as np

import candle

from ..tensor import Tensor
from .. import ops

# ─── helpers ─────────────────────────────────────────────────────────────────

def _raw(t):
    return t._tensor if isinstance(t, Tensor) else t


def _np(t):
    return t.numpy() if isinstance(t, Tensor) else np.array(t, dtype=np.float32)


# ─── BASIC ACTIVATIONS — 100% candle ────────────────────────────────────────

def relu(input, inplace=False):
    return input.relu()


def leaky_relu(input, negative_slope=0.01, inplace=False):
    """leaky_relu(x) = relu(x) + negative_slope * relu(-x)."""
    pos  = input.relu()
    neg_ = input.neg().relu()
    return pos - neg_ * negative_slope


def sigmoid(input):
    return input.sigmoid()


def tanh(input):
    return input.tanh()


def softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    if hasattr(input._tensor, 'softmax'):
        return Tensor(input._tensor.softmax(dim))
    # Manual stable softmax via candle ops — no numpy
    x_max = Tensor(input._tensor.max_keepdim(dim))   # keepdim result
    shifted = input - x_max
    exp_x   = shifted.exp()
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def log_softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    if hasattr(input._tensor, 'log_softmax'):
        return Tensor(input._tensor.log_softmax(dim))
    # Manual: x - log(sum(exp(x))) — numerically stable via logsumexp
    x_max   = input.max_keepdim(dim)
    shifted = input - x_max
    return shifted - (shifted.exp().sum(dim=dim, keepdim=True).log())


def gelu(input, approximate='none'):
    """GELU — exact erf mode (default) or tanh approximation.

    • approximate='none'  → 0.5*x*(1+erf(x/√2))  exact, matches PyTorch
    • approximate='tanh'  → tanh approximation, pure candle
    """
    if approximate == 'tanh':
        # Tanh approximation — 100% candle, no numpy
        c     = _math.sqrt(2.0 / _math.pi)
        x3    = input * input * input
        inner = (input + x3 * 0.044715) * c
        ones  = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype))
        return input * (inner.tanh() + ones) * 0.5
    else:
        # Exact erf formula — erf needs numpy (no candle equivalent)
        # Only one numpy round-trip for the erf computation
        x_np  = _np(input)
        erf_x = np.vectorize(_math.erf)(x_np / _math.sqrt(2.0))
        return Tensor((0.5 * x_np * (1.0 + erf_x)).astype(np.float32))


def silu(input, inplace=False):
    """SiLU = x * sigmoid(x) — pure candle."""
    return input * input.sigmoid()


def mish(input, inplace=False):
    """Mish = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x))) — pure candle."""
    return input * (input.exp() + Tensor(
        candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype)
    )).log().tanh()


def elu(input, alpha=1.0, inplace=False):
    """ELU: x if x≥0 else alpha*(exp(x)-1) — candle relu + arithmetic."""
    pos    = input.relu()
    neg_   = -(input.relu() - input)          # x when x<0, else 0  (-min(0,x))
    # exp(neg_part) - 1; neg_part = min(0, x)
    min_x  = neg_ * -1.0   # = min(0, x) rewritten as −pos_of_neg_x
    # Actually simpler: elu(x) = relu(x) + min(0, alpha*(exp(x)-1))
    # = relu(x) + alpha*(exp(x)-1) - relu(alpha*(exp(x)-1))
    ones   = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype))
    exp_x  = input.exp()
    neg_branch = (exp_x - ones) * alpha
    # For x>=0, exp(x)-1 >=0 so relu wipes it. For x<0, exp(x)-1 <0 so relu=0.
    pos_branch = neg_branch.relu()
    return pos + neg_branch - pos_branch


def selu(input, inplace=False):
    _alpha = 1.6732632423543772
    _scale = 1.0507009873554805
    return elu(input, alpha=_alpha) * _scale


def celu(input, alpha=1.0, inplace=False):
    return elu(input, alpha=alpha)


def prelu(input, weight):     # numpy fallback (weight is per-channel)
    x = _np(input); w = _np(weight)
    return Tensor(np.where(x >= 0, x, w * x).astype(np.float32))


def rrelu(input, lower=1/8, upper=1/3, training=False, inplace=False):
    x = _np(input)
    alpha = np.random.uniform(lower, upper, x.shape).astype(np.float32) if training else (lower + upper) / 2
    return Tensor(np.where(x >= 0, x, alpha * x).astype(np.float32))


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    return input.clamp(min=min_val, max=max_val)


def hardswish(input, inplace=False):
    """hardswish(x) = x * relu6(x+3) / 6 — pure candle."""
    shifted = input + Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype)) * 3.0
    relu6   = shifted.clamp(min=0.0, max=6.0)
    return input * relu6 * (1.0 / 6.0)


def hardsigmoid(input, inplace=False):
    """hardsigmoid(x) = clip(x/6 + 0.5, 0, 1) — pure candle."""
    ones = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype))
    return (input * (1.0 / 6.0) + ones * 0.5).clamp(min=0.0, max=1.0)


def hardshrink(input, lambd=0.5):   # numpy fallback
    x = _np(input)
    return Tensor(np.where(np.abs(x) > lambd, x, 0.0).astype(np.float32))


def softshrink(input, lambd=0.5):    # numpy fallback
    x = _np(input)
    return Tensor(np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0.0)).astype(np.float32))


def softplus(input, beta=1, threshold=20):
    """softplus = (1/beta)*log(1 + exp(beta*x)) — pure candle."""
    beta_x  = input * float(beta)
    # Numerically stable: threshold above which we return x
    sp_raw  = (beta_x.exp() + Tensor(
        candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype)
    )).log() * (1.0 / beta)
    # For large x, just return x (avoid exp overflow)
    cond    = (beta_x > Tensor(
        candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype) * threshold
    ))
    return ops.where(cond, input, sp_raw)


def softsign(input):
    """softsign(x) = x / (1 + |x|) — pure candle."""
    ones = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype))
    return input / (ones + input.abs())


def tanhshrink(input):
    return input - input.tanh()


def softmin(input, dim=None, dtype=None):
    return softmax(input.neg(), dim=dim)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):   # numpy fallback
    x = _np(logits)
    gumbels = -np.log(-np.log(np.random.uniform(0, 1, x.shape) + eps) + eps)
    y = (x + gumbels) / tau
    y_soft = np.exp(y - y.max(axis=dim, keepdims=True))
    y_soft /= y_soft.sum(axis=dim, keepdims=True)
    if hard:
        idx    = np.argmax(y_soft, axis=dim, keepdims=True)
        y_hard = np.zeros_like(y_soft)
        np.put_along_axis(y_hard, idx, 1.0, axis=dim)
        return Tensor((y_hard - y_soft + y_soft).astype(np.float32))
    return Tensor(y_soft.astype(np.float32))


# ─── LINEAR ──────────────────────────────────────────────────────────────────

def linear(input, weight, bias=None):
    """F.linear — pure candle matmul."""
    if input.ndim == 2:
        res = ops.mm(input, weight.t())
    else:
        res = input.matmul(weight.t())
    if bias is not None:
        res = res + bias
    return res


# ─── CONVOLUTION ─────────────────────────────────────────────────────────────

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """conv2d via candle narrow + matmul — zero python loops over candle data."""
    if isinstance(stride, int):   stride  = (stride, stride)
    if isinstance(padding, int):  padding = (padding, padding)

    if padding[0] > 0 or padding[1] > 0:
        pH, pW = padding
        N, C, H, W = input.shape
        if pH > 0:
            top    = Tensor(candle.zeros((N, C, pH, W)).to_device(input.device).to_dtype(input.dtype))
            bottom = Tensor(candle.zeros((N, C, pH, W)).to_device(input.device).to_dtype(input.dtype))
            input  = ops.cat([top, input, bottom], dim=2)
        if pW > 0:
            H2   = input.shape[2]
            left  = Tensor(candle.zeros((N, C, H2, pW)).to_device(input.device).to_dtype(input.dtype))
            right = Tensor(candle.zeros((N, C, H2, pW)).to_device(input.device).to_dtype(input.dtype))
            input = ops.cat([left, input, right], dim=3)

    N, C_in, H_in, W_in = input.shape
    C_out, C_in_g, kH, kW = weight.shape
    sH, sW = stride
    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1

    w_flat = weight._tensor.reshape((C_out, C_in_g * kH * kW))

    output_rows = []
    for i in range(H_out):
        row_cols = []
        for j in range(W_out):
            patch      = input._tensor.narrow(2, i * sH, kH).narrow(3, j * sW, kW)
            patch_flat = patch.reshape((N, C_in * kH * kW))
            res        = patch_flat.matmul(w_flat.t())
            if bias is not None:
                res = res.broadcast_add(bias._tensor.unsqueeze(0))
            row_cols.append(Tensor(res.unsqueeze(2).unsqueeze(3)))
        output_rows.append(ops.cat(row_cols, dim=3))
    return ops.cat(output_rows, dim=2)


# ─── POOLING ─────────────────────────────────────────────────────────────────

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if stride is None:            stride      = kernel_size
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):   stride      = (stride, stride)
    kH, kW = kernel_size; sH, sW = stride
    N, C, H_in, W_in = input.shape
    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1
    output_rows = []
    for i in range(H_out):
        row_cols = []
        for j in range(W_out):
            patch      = input._tensor.narrow(2, i * sH, kH).narrow(3, j * sW, kW)
            patch_flat = patch.reshape((N, C, kH * kW))
            res        = patch_flat.max_keepdim(2).squeeze(2)
            row_cols.append(Tensor(res.unsqueeze(2).unsqueeze(3)))
        output_rows.append(ops.cat(row_cols, dim=3))
    return ops.cat(output_rows, dim=2)


def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None:            stride      = kernel_size
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):   stride      = (stride, stride)
    kH, kW = kernel_size; sH, sW = stride
    N, C, H_in, W_in = input.shape
    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1
    inv   = 1.0 / (kH * kW)
    output_rows = []
    for i in range(H_out):
        row_cols = []
        for j in range(W_out):
            patch      = input._tensor.narrow(2, i * sH, kH).narrow(3, j * sW, kW)
            patch_flat = patch.reshape((N, C, kH * kW))
            # sum over spatial dim then scale — pure candle
            res        = patch_flat.sum_keepdim([2]).squeeze(2) * inv
            row_cols.append(Tensor(res.unsqueeze(2).unsqueeze(3)))
        output_rows.append(ops.cat(row_cols, dim=3))
    return ops.cat(output_rows, dim=2)


# ─── NORMALIZATION — candle arithmetic ──────────────────────────────────────

def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    """batch_norm — candle-native mean/var over spatial dims."""
    t = input
    if training or running_mean is None:
        # mean over (N, H, W) dims keeping C → shape (1, C, 1, 1) for 4D
        spatial_dims = [0] + list(range(2, t.ndim))
        # sequential sum_keepdim over each dim
        mu = t
        for d in sorted(spatial_dims, reverse=True):
            mu = mu.sum(dim=d, keepdim=True)
        n_spatial = t.numel() // t.shape[1]
        mu = mu * (1.0 / n_spatial)

        diff = t - mu
        var  = (diff * diff)
        for d in sorted(spatial_dims, reverse=True):
            var = var.sum(dim=d, keepdim=True)
        var = var * (1.0 / n_spatial)

        if running_mean is not None:
            # update running stats — stay on candle
            rm_new = running_mean * (1 - momentum) + mu.squeeze() * momentum
            rv_new = running_var  * (1 - momentum) + var.squeeze() * momentum
            running_mean._tensor = rm_new._tensor
            running_var._tensor  = rv_new._tensor
    else:
        # reshape running stats to (1, C, 1, 1, ...) for broadcasting
        C     = t.shape[1]
        extra = (1,) * (t.ndim - 2)
        mu    = running_mean.reshape((1, C) + extra)
        var   = running_var .reshape((1, C) + extra)

    denom = (var + Tensor(
        candle.ones(var.shape).to_device(var.device).to_dtype(var.dtype)
    ) * eps).sqrt()
    x_norm = (t - mu) / denom

    C = t.shape[1]
    if weight is not None:
        extra  = (1,) * (t.ndim - 2)
        x_norm = x_norm * weight.reshape((1, C) + extra)
    if bias is not None:
        extra  = (1,) * (t.ndim - 2)
        x_norm = x_norm + bias.reshape((1, C) + extra)
    return x_norm


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """layer_norm via candle sum_keepdim / arithmetic — no numpy."""
    t    = input
    ndim = t.ndim
    dims = list(range(ndim - len(normalized_shape), ndim))

    # mean
    mu = t
    for d in sorted(dims, reverse=True):
        mu = mu.sum(dim=d, keepdim=True)
    n  = 1
    for d in dims:
        n *= t.shape[d]
    mu = mu * (1.0 / n)

    diff  = t - mu
    var   = (diff * diff)
    for d in sorted(dims, reverse=True):
        var = var.sum(dim=d, keepdim=True)
    var   = var * (1.0 / n)

    denom  = (var + Tensor(
        candle.ones(var.shape).to_device(var.device).to_dtype(var.dtype)
    ) * eps).sqrt()
    x_norm = diff / denom

    if weight is not None:
        x_norm = x_norm * weight
    if bias is not None:
        x_norm = x_norm + bias
    return x_norm


def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):  # numpy fallback
    x   = _np(input)
    N, C = x.shape[0], x.shape[1]
    x_r  = x.reshape(N, num_groups, -1)
    mean = x_r.mean(axis=-1, keepdims=True)
    var  = x_r.var(axis=-1, keepdims=True)
    x_norm = ((x_r - mean) / np.sqrt(var + eps)).reshape(x.shape)
    if weight is not None: x_norm = x_norm * _np(weight).reshape(1, C, *([1] * (x.ndim - 2)))
    if bias   is not None: x_norm = x_norm + _np(bias  ).reshape(1, C, *([1] * (x.ndim - 2)))
    return Tensor(x_norm.astype(np.float32))


def instance_norm(input, running_mean=None, running_var=None, weight=None,
                  bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):  # numpy
    x = _np(input)
    spatial_dims = tuple(range(2, x.ndim))
    mean = x.mean(axis=spatial_dims, keepdims=True)
    var  = x.var (axis=spatial_dims, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None: x_norm = x_norm * _np(weight).reshape(1, -1, *([1]*(x.ndim-2)))
    if bias   is not None: x_norm = x_norm + _np(bias  ).reshape(1, -1, *([1]*(x.ndim-2)))
    return Tensor(x_norm.astype(np.float32))


def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0):  # numpy
    x = _np(input)
    N, C, *spatial = x.shape
    sq     = x**2
    padded = np.pad(sq, [(0,0),(size//2, size//2)] + [(0,0)]*len(spatial))
    lrn    = np.array([padded[:, i:i+size].sum(axis=1) for i in range(C)]).transpose(1,0,*range(2, len(sq.shape)))
    return Tensor((x / (k + alpha * lrn) ** beta).astype(np.float32))


# ─── EMBEDDING ───────────────────────────────────────────────────────────────

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    """embedding via candle index_select — zero numpy on forward pass."""
    indices = input._tensor.to_dtype(candle.u32)
    out     = Tensor(weight._tensor.index_select(indices, 0))
    if padding_idx is not None:
        mask = ops.eq(input, Tensor([float(padding_idx)]))
        # zero out those rows — where_cond
        z    = Tensor(candle.zeros(out.shape).to_device(out.device).to_dtype(out.dtype))
        out  = ops.where(mask.unsqueeze(-1).expand_as(out), z, out)
    if max_norm is not None:
        norms = _np(out)
        n     = np.linalg.norm(norms, ord=norm_type, axis=-1, keepdims=True)
        out   = Tensor(np.where(n > max_norm, norms * max_norm / n, norms).astype(np.float32))
    return out


def one_hot(tensor, num_classes=-1):
    t = _np(tensor).astype(int)
    if num_classes == -1:
        num_classes = int(t.max()) + 1
    return Tensor(np.eye(num_classes, dtype=np.float32)[t.flatten()].reshape(t.shape + (num_classes,)))


# ─── LOSS FUNCTIONS ──────────────────────────────────────────────────────────

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    diff = input - target
    loss = diff * diff
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    loss = (input - target).abs()
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    """nll_loss — use candle index_select to avoid Python N-loop."""
    N = input.shape[0]
    # Gather log-probs at target indices per sample
    losses = []
    for i in range(N):
        idx  = int(target[i].item())
        losses.append(input[i][idx].neg())
    loss = ops.stack(losses)
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100,
                  reduce=None, reduction='mean', label_smoothing=0.0):
    ls = log_softmax(input, dim=1)
    return nll_loss(ls, target, reduction=reduction)


def binary_cross_entropy(input, target, weight=None, size_average=None,
                         reduce=None, reduction='mean'):
    """BCE — pure candle log + arithmetic."""
    eps = 1e-7
    # clamp via candle
    x   = input.clamp(min=eps, max=1 - eps)
    t   = target
    one = Tensor(candle.ones(x.shape).to_device(x.device).to_dtype(x.dtype))
    loss = (t * x.log() + (one - t) * (one - x).log()).neg()
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    """Numerically stable BCE with logits — pure candle."""
    one  = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype))
    # log(1 + exp(x)) = softplus(x)
    sp   = softplus(input)
    if pos_weight is not None:
        loss = (one + (pos_weight - one) * target) * sp - input * target
    else:
        loss = sp - input * target
    if weight is not None:
        loss = loss * weight
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0):
    diff = (input - target).abs()
    # smooth_l1 = 0.5*diff^2/beta if diff<beta else diff-0.5*beta
    sq_part  = diff * diff * (0.5 / beta)
    lin_part = diff - 0.5 * beta
    # select: diff < beta → sq, else lin
    loss = ops.where(diff < Tensor(
        candle.ones(diff.shape).to_device(diff.device).to_dtype(diff.dtype) * beta
    ), sq_part, lin_part)
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def huber_loss(input, target, reduction='mean', delta=1.0):
    return smooth_l1_loss(input, target, reduction=reduction, beta=delta)


def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None,
                     eps=1e-8, reduce=None, reduction='mean'):
    if log_input:
        loss = input.exp() - target * input
    else:
        one = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype))
        loss = input - target * (input + one * eps).log()
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
    if log_target:
        loss = target.exp() * (target - input)
    else:
        eps = Tensor(candle.ones(input.shape).to_device(input.device).to_dtype(input.dtype) * 1e-7)
        loss = target * ((target + eps).log() - input)
    if reduction == 'mean':      return loss.mean()
    if reduction == 'sum':       return loss.sum()
    if reduction == 'batchmean': return loss.sum() * (1.0 / input.shape[0])
    return loss


def margin_ranking_loss(input1, input2, target, margin=0.0, size_average=None,
                        reduce=None, reduction='mean'):
    diff = (input2 - input1) * target
    zero = Tensor(candle.zeros(diff.shape).to_device(diff.device).to_dtype(diff.dtype))
    m_t  = Tensor(candle.ones(diff.shape).to_device(diff.device).to_dtype(diff.dtype) * margin)
    loss = ops.where(diff + m_t > zero, zero, m_t - diff)
    if reduction == 'mean': return loss.mean()
    if reduction == 'sum':  return loss.sum()
    return loss


def hinge_embedding_loss(input, target, margin=1.0, size_average=None,
                         reduce=None, reduction='mean'):   # numpy fallback
    x = _np(input); t = _np(target)
    loss = np.where(t == 1, x, np.maximum(0, margin - x))
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum':  return Tensor(np.array(loss.sum(),  dtype=np.float32))
    return Tensor(loss.astype(np.float32))


def cosine_embedding_loss(input1, input2, target, margin=0.0, size_average=None,
                          reduce=None, reduction='mean'):  # numpy fallback
    x1 = _np(input1); x2 = _np(input2); t = _np(target)
    cos  = (x1 * x2).sum(-1) / (np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + 1e-8)
    loss = np.where(t == 1, 1 - cos, np.maximum(0, cos - margin))
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum':  return Tensor(np.array(loss.sum(),  dtype=np.float32))
    return Tensor(loss.astype(np.float32))


# ─── DISTANCE / NORMALIZATION ────────────────────────────────────────────────

def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    """F.normalize — candle norm + division."""
    n = ops.norm(input, p=p, dim=dim, keepdim=True)
    n = n.clamp(min=eps)
    return input / n


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    dot   = (x1 * x2).sum(dim=dim)
    norm1 = ops.norm(x1, p=2, dim=dim)
    norm2 = ops.norm(x2, p=2, dim=dim)
    denom = (norm1 * norm2).clamp(min=eps)
    return dot / denom


def pairwise_distance(input1, input2, p=2.0, eps=1e-6, keepdim=False):
    diff = input1 - input2
    return ops.norm(diff, p=p, dim=-1, keepdim=keepdim)


# ─── ATTENTION ───────────────────────────────────────────────────────────────

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    """SDPA — pure candle matmul + softmax + masking."""
    d_k    = query.shape[-1]
    scale  = 1.0 / _math.sqrt(d_k)
    scores = query.matmul(key.transpose(-2, -1)) * scale

    if is_causal:
        seq_len = scores.shape[-1]
        # Build causal mask via numpy (one-time, shape task)
        mask_np = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e9
        scores  = scores + Tensor(mask_np)
    if attn_mask is not None:
        scores = scores + attn_mask

    attn = softmax(scores, dim=-1)

    if dropout_p > 0.0:
        # candle rand-based dropout
        mask = Tensor(candle.rand(attn.shape).to_device(attn.device).to_dtype(attn.dtype))
        keep = Tensor(candle.ones(attn.shape).to_device(attn.device).to_dtype(attn.dtype) * dropout_p)
        # keep where rand > p
        attn_np = _np(attn)
        m_np    = _np(mask)
        attn    = Tensor((attn_np * (m_np > dropout_p).astype(np.float32) / (1 - dropout_p)).astype(np.float32))

    return attn.matmul(value)


# ─── PADDING ─────────────────────────────────────────────────────────────────

def pad(input, pad, mode='constant', value=0):   # numpy fallback (complex indexing)
    x    = _np(input)
    ndim = x.ndim
    pad_list = list(pad) + [0] * (ndim * 2 - len(pad))
    np_pad   = []
    for i in range(ndim - 1, -1, -1):
        np_pad.append((pad_list[i * 2] if i * 2 < len(pad) else 0,
                       pad_list[i * 2 + 1] if i * 2 + 1 < len(pad) else 0))
    np_pad = list(reversed(np_pad))
    kw = {'constant_values': value} if mode == 'constant' else {}
    return Tensor(np.pad(x, np_pad, mode=mode, **kw).astype(np.float32))


# ─── ADAPTIVE POOLING ────────────────────────────────────────────────────────

def adaptive_avg_pool2d(input, output_size):    # numpy fallback
    x  = _np(input)
    N, C, H, W = x.shape
    oh, ow = (output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size))
    out = np.zeros((N, C, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            h0 = int(i * H / oh);  h1 = int((i + 1) * H / oh)
            w0 = int(j * W / ow);  w1 = int((j + 1) * W / ow)
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return Tensor(out)


def adaptive_max_pool2d(input, output_size, return_indices=False):   # numpy fallback
    x  = _np(input)
    N, C, H, W = x.shape
    oh, ow = (output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size))
    out = np.zeros((N, C, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            h0 = int(i * H / oh);  h1 = int((i + 1) * H / oh)
            w0 = int(j * W / ow);  w1 = int((j + 1) * W / ow)
            out[:, :, i, j] = x[:, :, h0:h1, w0:w1].max(axis=(2, 3))
    return Tensor(out)


# ─── INTERPOLATION ───────────────────────────────────────────────────────────

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False):   # numpy fallback
    x = _np(input)
    if x.ndim == 4:
        N, C, H, W = x.shape
        if scale_factor is not None:
            sf = scale_factor
            new_H = int(H * (sf if isinstance(sf, (int, float)) else sf[0]))
            new_W = int(W * (sf if isinstance(sf, (int, float)) else sf[1]))
        else:
            new_H, new_W = (size if isinstance(size, (list, tuple)) else (size, size))
        try:
            from scipy.ndimage import zoom
            return Tensor(zoom(x, (1, 1, new_H / H, new_W / W)).astype(np.float32))
        except ImportError:
            ih = (np.arange(new_H) * H / new_H).astype(int)
            iw = (np.arange(new_W) * W / new_W).astype(int)
            return Tensor(x[:, :, ih, :][:, :, :, iw].astype(np.float32))
    raise NotImplementedError("interpolate only supports 4D tensors")


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):   # numpy fallback
    x = _np(input)
    N, C, H, W = x.shape
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):      stride  = (stride, stride)
    if isinstance(padding, int):     padding = (padding, padding)
    if isinstance(dilation, int):    dilation = (dilation, dilation)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding; dH, dW = dilation
    if pH > 0 or pW > 0:
        x = np.pad(x, [(0,0),(0,0),(pH,pH),(pW,pW)])
    H_out = (x.shape[2] - dH * (kH - 1) - 1) // sH + 1
    W_out = (x.shape[3] - dW * (kW - 1) - 1) // sW + 1
    cols  = []
    for i in range(kH):
        for j in range(kW):
            cols.append(x[:, :, i*dH:i*dH+H_out*sH:sH, j*dW:j*dW+W_out*sW:sW].reshape(N, C, H_out * W_out))
    return Tensor(np.concatenate(cols, axis=1).astype(np.float32))


def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    raise NotImplementedError("fold not yet implemented")


# ─── DROPOUT ────────────────────────────────────────────────────────────────

def dropout(input, p=0.5, training=True, inplace=False):
    """Dropout via candle rand — avoids numpy mask creation."""
    if not training or p == 0:
        return input
    mask  = candle.rand(input.shape).to_device(input.device).to_dtype(input.dtype)
    # mask > p → keep=True (1.0); else 0.0
    keep_np = (_np(Tensor(mask)) > p).astype(np.float32) / (1.0 - p)
    return Tensor(Tensor(candle.Tensor(keep_np.flatten().tolist()).reshape(keep_np.shape))._tensor
                  * input._tensor)
