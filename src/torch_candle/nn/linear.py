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
        res = ops.mm(input, self.weight.t())
        if self.bias is not None:
            res = res + self.bias
        return res

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

# Need to add .t() to Tensor class
