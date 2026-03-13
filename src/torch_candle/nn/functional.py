try:
    import candle
except ImportError:
    candle = None

from ..tensor import Tensor
from .. import ops

def relu(input, inplace=False):
    if not hasattr(input, '_tensor'):
        raise TypeError(f"Expected Tensor, got {type(input)}")
    return input.relu()

def leaky_relu(input, negative_slope=0.01, inplace=False):
    # leaky_relu = max(0, x) + min(0, x) * slope
    # Or in candle: relu(x) - relu(-x) * slope
    if not hasattr(input, 'relu'):
        raise TypeError(f"Expected Tensor, got {type(input)}")
    return input.relu() - input.neg().relu() * negative_slope

def sigmoid(input):
    return input.sigmoid()

def tanh(input):
    return input.tanh()

def softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    return Tensor(input._tensor.softmax(dim))

def log_softmax(input, dim=None, dtype=None):
    if dim is None:
        dim = -1
    # Fallback log_softmax if missing
    if hasattr(input._tensor, 'log_softmax'):
        return Tensor(input._tensor.log_softmax(dim))
    # log_softmax(x) = x - log(sum(exp(x)))
    x_max = input.max_keepdim(dim)
    shifted = input - x_max
    return shifted - (shifted.exp().sum(dim=dim, keepdim=True).log())

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # Fallback implementation of conv2d using narrow and matmul
    # input: (N, C_in, H_in, W_in)
    # weight: (C_out, C_in/groups, kH, kW)
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    
    # Handle padding (manual implementation using cat)
    if padding[0] > 0 or padding[1] > 0:
        pH, pW = padding
        N, C, H, W = input.shape
        if pH > 0:
            top = Tensor(candle.zeros((N, C, pH, W)).to_device(input.device).to_dtype(input.dtype))
            bottom = Tensor(candle.zeros((N, C, pH, W)).to_device(input.device).to_dtype(input.dtype))
            input = ops.cat([top, input, bottom], dim=2)
            H += 2 * pH
        if pW > 0:
            left = Tensor(candle.zeros((N, C, H, pW)).to_device(input.device).to_dtype(input.dtype))
            right = Tensor(candle.zeros((N, C, H, pW)).to_device(input.device).to_dtype(input.dtype))
            input = ops.cat([left, input, right], dim=3)

    N, C_in, H_in, W_in = input.shape
    C_out, C_in_g, kH, kW = weight.shape
    sH, sW = stride
    
    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1
    
    # We want to perform the convolution. 
    # Simplified approach: loop over H_out, W_out and extract patches
    # Then matmul with flattened kernels
    
    # weight_reshaped: (C_out, C_in_g * kH * kW)
    w_flat = weight._tensor.reshape((C_out, C_in_g * kH * kW))
    
    output_rows = []
    for i in range(H_out):
        row_cols = []
        for j in range(W_out):
            # Extract patch: (N, C_in, kH, kW)
            patch = input._tensor.narrow(2, i * sH, kH).narrow(3, j * sW, kW)
            # Flatten patch: (N, C_in * kH * kW)
            patch_flat = patch.reshape((N, C_in * kH * kW))
            # Matrix multiply: (N, C_in * kH * kW) @ (C_in * kH * kW, C_out) -> (N, C_out)
            # res = patch_flat @ w_flat.t()
            res = patch_flat.matmul(w_flat.t())
            if bias is not None:
                res = res.broadcast_add(bias._tensor.unsqueeze(0))
            row_cols.append(res.unsqueeze(2).unsqueeze(3)) # (N, C_out, 1, 1)
        
        # Stack row columns along W dim
        output_rows.append(ops.cat([Tensor(p) for p in row_cols], dim=3))
        
    # Stack rows along H dim
    final_res = ops.cat([Tensor(r) for r in output_rows], dim=2)
    return final_res

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if stride is None: stride = kernel_size
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int): stride = (stride, stride)
    
    kH, kW = kernel_size
    sH, sW = stride
    N, C, H_in, W_in = input.shape
    
    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1
    
    output_rows = []
    for i in range(H_out):
        row_cols = []
        for j in range(W_out):
            # Extract patch: (N, C, kH, kW)
            patch = input._tensor.narrow(2, i * sH, kH).narrow(3, j * sW, kW)
            # Flatten spatially: (N, C, kH*kW)
            patch_flat = patch.reshape((N, C, kH * kW))
            # Max over dim 2: (N, C)
            res = patch_flat.max_keepdim(2).squeeze(2)
            row_cols.append(res.unsqueeze(2).unsqueeze(3))
        output_rows.append(ops.cat([Tensor(p) for p in row_cols], dim=3))
    
    return ops.cat([Tensor(r) for r in output_rows], dim=2)

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if stride is None: stride = kernel_size
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int): stride = (stride, stride)
    
    kH, kW = kernel_size
    sH, sW = stride
    N, C, H_in, W_in = input.shape
    
    H_out = (H_in - kH) // sH + 1
    W_out = (W_in - kW) // sW + 1
    
    output_rows = []
    for i in range(H_out):
        row_cols = []
        for j in range(W_out):
            patch = input._tensor.narrow(2, i * sH, kH).narrow(3, j * sW, kW)
            # Mean is a bit more complex in candle if not native
            # (N, C, kH*kW)
            patch_flat = patch.reshape((N, C, kH * kW))
            res = patch_flat.mean_all() # Note: mean_all is for whole tensor, this is WRONG
            # We need mean over dim 2. candle.Tensor has mean_all only?
            # Let's use sum / count
            res = patch_flat.sum_keepdim(2).squeeze(2) * (1.0 / (kH * kW))
            row_cols.append(res.unsqueeze(2).unsqueeze(3))
        output_rows.append(ops.cat([Tensor(p) for p in row_cols], dim=3))
    
    return ops.cat([Tensor(r) for r in output_rows], dim=2)

def linear(input, weight, bias=None):
    # input: (N, *, in_features)
    # weight: (out_features, in_features)
    # res = input @ weight.T + bias
    if input.ndim == 2:
        res = ops.mm(input, weight.t())
    else:
        # handle higher dims if candle matmul supports it
        res = input.matmul(weight.t())
        
    if bias is not None:
        res = res + bias
    return res

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    diff = input - target
    loss = diff * diff
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    # input: (N, C) (log-probabilities)
    # target: (N) (class indices)
    # loss = -input[range(N), target]
    
    # Since we don't have robust advanced indexing yet, we use a loop or gather if available
    # For now, a loop-based approach for small batches or gather if candle supports it
    N = input.shape[0]
    losses = []
    for i in range(N):
        idx = int(target[i].item())
        losses.append(-input[i, idx])
    
    from . import ops
    loss = ops.stack(losses)
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
    # cross_entropy = nll_loss(log_softmax(input), target)
    ls = log_softmax(input, dim=1)
    return nll_loss(ls, target, reduction=reduction)

def dropout(input, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return input
    if candle is None:
        raise ImportError("candle backend not available for dropout")
    # Simplified dropout masking
    import numpy as np
    mask = (np.random.random(input.shape) > p).astype(np.float32) / (1.0 - p)
    tensor_data = getattr(input, '_tensor')
    res = tensor_data * candle.Tensor(mask).to_device(input.device).to_dtype(input.dtype)
    return Tensor(res)

import numpy as np
import math as _math

# ============================================================
# Activation functions (Phase 3)
# ============================================================
def gelu(input, approximate='none'):
    x = input.numpy()
    if approximate == 'tanh':
        out = 0.5 * x * (1 + np.tanh(_math.sqrt(2.0 / _math.pi) * (x + 0.044715 * x**3)))
    else:
        # Use Python's math.erf for accuracy
        _erf_vec = np.vectorize(_math.erf)
        out = 0.5 * x * (1 + _erf_vec(x / _math.sqrt(2.0)))
    return Tensor(out.astype(np.float32))

def silu(input, inplace=False):
    x = input.numpy()
    return Tensor((x / (1 + np.exp(-x))).astype(np.float32))

def mish(input, inplace=False):
    x = input.numpy()
    return Tensor((x * np.tanh(np.log1p(np.exp(x)))).astype(np.float32))

def elu(input, alpha=1.0, inplace=False):
    x = input.numpy()
    return Tensor(np.where(x >= 0, x, alpha * (np.exp(x) - 1)).astype(np.float32))

def selu(input, inplace=False):
    alpha = 1.6732632423543772; scale = 1.0507009873554805
    x = input.numpy()
    return Tensor((scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))).astype(np.float32))

def celu(input, alpha=1.0, inplace=False):
    x = input.numpy()
    return Tensor(np.where(x >= 0, x, alpha * (np.exp(x / alpha) - 1)).astype(np.float32))

def prelu(input, weight):
    x = input.numpy(); w = weight.numpy()
    return Tensor(np.where(x >= 0, x, w * x).astype(np.float32))

def rrelu(input, lower=1/8, upper=1/3, training=False, inplace=False):
    x = input.numpy()
    if training:
        alpha = np.random.uniform(lower, upper, x.shape).astype(np.float32)
    else:
        alpha = (lower + upper) / 2
    return Tensor(np.where(x >= 0, x, alpha * x).astype(np.float32))

def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    return Tensor(np.clip(input.numpy(), min_val, max_val).astype(np.float32))

def hardswish(input, inplace=False):
    x = input.numpy()
    return Tensor((x * np.clip(x + 3, 0, 6) / 6).astype(np.float32))

def hardsigmoid(input, inplace=False):
    x = input.numpy()
    return Tensor(np.clip(x / 6 + 0.5, 0, 1).astype(np.float32))

def hardshrink(input, lambd=0.5):
    x = input.numpy()
    return Tensor(np.where(np.abs(x) > lambd, x, 0.0).astype(np.float32))

def softshrink(input, lambd=0.5):
    x = input.numpy()
    return Tensor(np.where(x > lambd, x - lambd, np.where(x < -lambd, x + lambd, 0.0)).astype(np.float32))

def softplus(input, beta=1, threshold=20):
    x = input.numpy()
    return Tensor(np.where(x * beta > threshold, x, (1 / beta) * np.log1p(np.exp(beta * x))).astype(np.float32))

def softsign(input):
    x = input.numpy()
    return Tensor((x / (1 + np.abs(x))).astype(np.float32))

def tanhshrink(input):
    x = input.numpy()
    return Tensor((x - np.tanh(x)).astype(np.float32))

def softmin(input, dim=None, dtype=None):
    return softmax(-input, dim=dim)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    x = logits.numpy()
    gumbels = -np.log(-np.log(np.random.uniform(0, 1, x.shape) + eps) + eps)
    y = (x + gumbels) / tau
    y_soft = np.exp(y - y.max(axis=dim, keepdims=True))
    y_soft /= y_soft.sum(axis=dim, keepdims=True)
    if hard:
        idx = np.argmax(y_soft, axis=dim, keepdims=True)
        y_hard = np.zeros_like(y_soft)
        np.put_along_axis(y_hard, idx, 1.0, axis=dim)
        return Tensor((y_hard - y_soft + y_soft).astype(np.float32))
    return Tensor(y_soft.astype(np.float32))

# ============================================================
# Normalization functions (Phase 3)
# ============================================================
def batch_norm(input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
    x = input.numpy()
    if training or running_mean is None:
        mean = x.mean(axis=(0, *range(2, x.ndim)), keepdims=True)
        var = x.var(axis=(0, *range(2, x.ndim)), keepdims=True)
        if running_mean is not None:
            rm = running_mean.numpy()
            rv = running_var.numpy()
            rm_new = (1 - momentum) * rm + momentum * mean.squeeze()
            rv_new = (1 - momentum) * rv + momentum * var.squeeze()
            running_mean._tensor = Tensor(rm_new.astype(np.float32))._tensor
            running_var._tensor = Tensor(rv_new.astype(np.float32))._tensor
    else:
        mean = running_mean.numpy().reshape((1, -1) + (1,) * (x.ndim - 2))
        var = running_var.numpy().reshape((1, -1) + (1,) * (x.ndim - 2))
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        w = weight.numpy().reshape((1, -1) + (1,) * (x.ndim - 2))
        x_norm = x_norm * w
    if bias is not None:
        b = bias.numpy().reshape((1, -1) + (1,) * (x.ndim - 2))
        x_norm = x_norm + b
    return Tensor(x_norm.astype(np.float32))

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    x = input.numpy()
    dims = tuple(range(-len(normalized_shape), 0))
    mean = x.mean(axis=dims, keepdims=True)
    var = x.var(axis=dims, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None: x_norm = x_norm * weight.numpy()
    if bias is not None: x_norm = x_norm + bias.numpy()
    return Tensor(x_norm.astype(np.float32))

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-5):
    x = input.numpy()
    N, C = x.shape[0], x.shape[1]
    x_r = x.reshape(N, num_groups, -1)
    mean = x_r.mean(axis=-1, keepdims=True)
    var = x_r.var(axis=-1, keepdims=True)
    x_norm = ((x_r - mean) / np.sqrt(var + eps)).reshape(x.shape)
    if weight is not None: x_norm = x_norm * weight.numpy().reshape(1, C, *([1] * (x.ndim - 2)))
    if bias is not None: x_norm = x_norm + bias.numpy().reshape(1, C, *([1] * (x.ndim - 2)))
    return Tensor(x_norm.astype(np.float32))

def instance_norm(input, running_mean=None, running_var=None, weight=None,
                  bias=None, use_input_stats=True, momentum=0.1, eps=1e-5):
    x = input.numpy()
    N, C = x.shape[:2]
    spatial_dims = tuple(range(2, x.ndim))
    mean = x.mean(axis=spatial_dims, keepdims=True)
    var = x.var(axis=spatial_dims, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if weight is not None: x_norm = x_norm * weight.numpy().reshape(1, -1, *([1]*(x.ndim-2)))
    if bias is not None: x_norm = x_norm + bias.numpy().reshape(1, -1, *([1]*(x.ndim-2)))
    return Tensor(x_norm.astype(np.float32))

def local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0):
    x = input.numpy()
    N, C, *spatial = x.shape
    sq = x**2
    padded = np.pad(sq, [(0,0),(size//2, size//2)] + [(0,0)]*len(spatial))
    lrn = np.array([padded[:, i:i+size].sum(axis=1) for i in range(C)]).transpose(1,0,*range(2, len(sq.shape)))
    return Tensor((x / (k + alpha * lrn) ** beta).astype(np.float32))

# ============================================================
# Sparse / Embedding functions (Phase 3)
# ============================================================
def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False):
    indices = input.numpy().astype(int)
    w = weight.numpy()
    out = w[indices]
    if padding_idx is not None:
        mask = indices == padding_idx
        out[mask] = 0.0
    if max_norm is not None:
        norms = np.linalg.norm(out, ord=norm_type, axis=-1, keepdims=True)
        out = np.where(norms > max_norm, out * max_norm / norms, out)
    return Tensor(out.astype(np.float32))

def one_hot(tensor, num_classes=-1):
    t = tensor.numpy().astype(int)
    if num_classes == -1: num_classes = int(t.max()) + 1
    out = np.eye(num_classes, dtype=np.float32)[t.flatten()].reshape(t.shape + (num_classes,))
    return Tensor(out)

# ============================================================
# Loss functions (Phase 3)
# ============================================================
def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    x = np.clip(input.numpy(), 1e-7, 1 - 1e-7)
    t = target.numpy()
    loss = -(t * np.log(x) + (1 - t) * np.log(1 - x))
    if weight is not None: loss = loss * weight.numpy()
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    x = input.numpy(); t = target.numpy()
    if pos_weight is not None:
        pw = pos_weight.numpy()
        loss = (1 - t) * x + (1 + (pw - 1) * t) * np.log1p(np.exp(-np.abs(x))) + np.maximum(-x, 0)
    else:
        loss = np.maximum(x, 0) - x * t + np.log1p(np.exp(-np.abs(x)))
    if weight is not None: loss = loss * weight.numpy()
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    loss = np.abs(input.numpy() - target.numpy())
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean', beta=1.0):
    diff = np.abs(input.numpy() - target.numpy())
    loss = np.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def huber_loss(input, target, reduction='mean', delta=1.0):
    return smooth_l1_loss(input, target, reduction=reduction, beta=delta)

def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None,
                     eps=1e-8, reduce=None, reduction='mean'):
    x = input.numpy(); t = target.numpy()
    if log_input:
        loss = np.exp(x) - t * x
    else:
        loss = x - t * np.log(x + eps)
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False):
    x = input.numpy(); t = target.numpy()
    if log_target:
        loss = np.exp(t) * (t - x)
    else:
        loss = t * (np.log(t + 1e-7) - x)
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    if reduction == 'batchmean': return Tensor(np.array(loss.sum() / x.shape[0], dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def margin_ranking_loss(input1, input2, target, margin=0.0, size_average=None,
                        reduce=None, reduction='mean'):
    x1 = input1.numpy(); x2 = input2.numpy(); t = target.numpy()
    loss = np.maximum(0, -t * (x1 - x2) + margin)
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def hinge_embedding_loss(input, target, margin=1.0, size_average=None,
                         reduce=None, reduction='mean'):
    x = input.numpy(); t = target.numpy()
    loss = np.where(t == 1, x, np.maximum(0, margin - x))
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

def cosine_embedding_loss(input1, input2, target, margin=0.0, size_average=None,
                          reduce=None, reduction='mean'):
    x1 = input1.numpy(); x2 = input2.numpy(); t = target.numpy()
    cos = (x1 * x2).sum(-1) / (np.linalg.norm(x1, axis=-1) * np.linalg.norm(x2, axis=-1) + 1e-8)
    loss = np.where(t == 1, 1 - cos, np.maximum(0, cos - margin))
    if reduction == 'mean': return Tensor(np.array(loss.mean(), dtype=np.float32))
    if reduction == 'sum': return Tensor(np.array(loss.sum(), dtype=np.float32))
    return Tensor(loss.astype(np.float32))

# ============================================================
# Vision / distance operations
# ============================================================
def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    x = input.numpy()
    n = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    return Tensor((x / np.maximum(n, eps)).astype(np.float32))

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    a = x1.numpy(); b = x2.numpy()
    dot = (a * b).sum(axis=dim)
    norm = np.linalg.norm(a, axis=dim) * np.linalg.norm(b, axis=dim)
    return Tensor((dot / np.maximum(norm, eps)).astype(np.float32))

def pairwise_distance(input1, input2, p=2.0, eps=1e-6, keepdim=False):
    diff = input1.numpy() - input2.numpy()
    dist = np.linalg.norm(diff + eps, ord=p, axis=-1)
    return Tensor(dist.astype(np.float32))

def pad(input, pad, mode='constant', value=0):
    x = input.numpy()
    # pad is (left, right, top, bottom, ...) in reverse dim order for PyTorch
    # numpy pad takes (before, after) tuples left-to-right
    ndim = x.ndim
    if isinstance(pad, int): pad = [pad]
    pairs = [(0, 0)] * ndim
    for i, (p_b, p_a) in enumerate(zip(pad[1::2], pad[::2])):
        pairs[ndim - 1 - i // 2] = (p_b if i % 2 == 0 else 0, p_a if i % 2 == 1 else 0)
    # Simplified: just use np.pad
    np_pad = []
    pad_list = list(pad) + [0] * (ndim * 2 - len(pad))
    for i in range(ndim - 1, -1, -1):
        np_pad.append((pad_list[i * 2] if i * 2 < len(pad) else 0,
                       pad_list[i * 2 + 1] if i * 2 + 1 < len(pad) else 0))
    np_pad = list(reversed(np_pad))
    kwargs = {} if mode == 'constant' else {}
    return Tensor(np.pad(x, np_pad, mode=mode if mode != 'reflect' else 'reflect',
                         **({'constant_values': value} if mode == 'constant' else {})).astype(np.float32))

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None,
                recompute_scale_factor=None, antialias=False):
    x = input.numpy()
    if x.ndim == 4:
        N, C, H, W = x.shape
        if scale_factor is not None:
            if isinstance(scale_factor, (int, float)):
                new_H, new_W = int(H * scale_factor), int(W * scale_factor)
            else:
                new_H, new_W = int(H * scale_factor[0]), int(W * scale_factor[1])
        else:
            new_H, new_W = (size if isinstance(size, (list, tuple)) else (size, size))
        try:
            from scipy.ndimage import zoom
            scale = (1, 1, new_H / H, new_W / W)
            return Tensor(zoom(x, scale).astype(np.float32))
        except ImportError:
            # nearest fallback
            idx_h = (np.arange(new_H) * H / new_H).astype(int)
            idx_w = (np.arange(new_W) * W / new_W).astype(int)
            return Tensor(x[:, :, idx_h, :][:, :, :, idx_w].astype(np.float32))
    raise NotImplementedError(f"interpolate only supports 4D tensors")

def adaptive_avg_pool2d(input, output_size):
    x = input.numpy()
    N, C, H, W = x.shape
    oh, ow = (output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size))
    out = np.zeros((N, C, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            h_start = int(i * H / oh); h_end = int((i + 1) * H / oh)
            w_start = int(j * W / ow); w_end = int((j + 1) * W / ow)
            out[:, :, i, j] = x[:, :, h_start:h_end, w_start:w_end].mean(axis=(2, 3))
    return Tensor(out)

def adaptive_max_pool2d(input, output_size, return_indices=False):
    x = input.numpy()
    N, C, H, W = x.shape
    oh, ow = (output_size if isinstance(output_size, (list, tuple)) else (output_size, output_size))
    out = np.zeros((N, C, oh, ow), dtype=np.float32)
    for i in range(oh):
        for j in range(ow):
            h_start = int(i * H / oh); h_end = int((i + 1) * H / oh)
            w_start = int(j * W / ow); w_end = int((j + 1) * W / ow)
            out[:, :, i, j] = x[:, :, h_start:h_end, w_start:w_end].max(axis=(2, 3))
    return Tensor(out)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    q = query.numpy(); k = key.numpy(); v = value.numpy()
    d_k = q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
    if is_causal:
        seq_len = scores.shape[-1]
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    if attn_mask is not None:
        scores = scores + attn_mask.numpy()
    attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn /= attn.sum(axis=-1, keepdims=True)
    if dropout_p > 0:
        mask = (np.random.random(attn.shape) > dropout_p).astype(np.float32) / (1 - dropout_p)
        attn = attn * mask
    return Tensor(np.matmul(attn, v).astype(np.float32))

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    x = input.numpy()
    N, C, H, W = x.shape
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    kH, kW = kernel_size; sH, sW = stride; pH, pW = padding; dH, dW = dilation
    if pH > 0 or pW > 0:
        x = np.pad(x, [(0,0),(0,0),(pH,pH),(pW,pW)])
    H_out = (x.shape[2] - dH * (kH - 1) - 1) // sH + 1
    W_out = (x.shape[3] - dW * (kW - 1) - 1) // sW + 1
    cols = []
    for i in range(kH):
        for j in range(kW):
            cols.append(x[:, :, i*dH:i*dH+H_out*sH:sH, j*dW:j*dW+W_out*sW:sW].reshape(N, C, H_out * W_out))
    return Tensor(np.concatenate(cols, axis=1).astype(np.float32))

def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    raise NotImplementedError("fold not yet implemented — use Tensor operations directly")

