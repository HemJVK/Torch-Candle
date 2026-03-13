"""torch_candle.nn.modules.normalization — Normalization layers"""
import numpy as np
import math
from ..module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init
from .. import functional as F

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Parameter(Tensor(np.ones(num_features, dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(num_features, dtype=np.float32)))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', Tensor(np.zeros(num_features, dtype=np.float32)))
            self.register_buffer('running_var', Tensor(np.ones(num_features, dtype=np.float32)))
            self.register_buffer('num_batches_tracked', Tensor(np.array(0, dtype=np.int64)))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, input):
        # We manually apply batch norm using numpy for now, will dispatch to F.batch_norm
        return F.batch_norm(
            input,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            self.weight, self.bias,
            self.training or not self.track_running_stats,
            self.momentum, self.eps
        )

class BatchNorm2d(BatchNorm1d):
    pass

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(Tensor(np.ones(self.normalized_shape, dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(self.normalized_shape, dtype=np.float32)))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = Parameter(Tensor(np.ones(num_channels, dtype=np.float32)))
            self.bias = Parameter(Tensor(np.zeros(num_channels, dtype=np.float32)))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)
