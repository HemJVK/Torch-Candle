import math
from .module import Module
from .parameter import Parameter
from .. import ops

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use our own factory methods
        from .. import randn, zeros
        k = math.sqrt(1.0 / in_features)
        
        # Initialize weights and scale
        w_data = (randn(out_features, in_features) * (2 * k)) - k
        self.weight = Parameter(w_data)
        
        if bias:
            b_data = (randn(out_features) * (2 * k)) - k
            self.bias = Parameter(b_data)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        orig_shape = input.shape
        if len(orig_shape) > 2:
            import numpy as np
            in_features = orig_shape[-1]
            batch_dims = int(np.prod(orig_shape[:-1]))
            x_2d = input.view(batch_dims, in_features)
            res_2d = ops.mm(x_2d, self.weight.t())
            if self.bias is not None:
                res_2d = res_2d + self.bias
            out_shape = list(orig_shape[:-1]) + [self.out_features]
            return res_2d.view(*out_shape)
        else:
            res = ops.mm(input, self.weight.t())
            if self.bias is not None:
                res = res + self.bias
            return res

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

# Need to add .t() to Tensor class
