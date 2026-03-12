import math
from .module import Module
from .parameter import Parameter
from .. import ops

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights matching PyTorch's kaiming_uniform_
        # For simplicity, using a uniform distribution here
        k = 1.0 / math.sqrt(in_features)
        
        # Use ones/randn and then scale
        # Candle doesn't have uniform easily, let's use randn for now or just scale rand
        import candle
        w_data = candle.randn((out_features, in_features)) * k
        self.weight = Parameter(w_data)
        
        if bias:
            b_data = candle.randn((out_features,)) * k
            self.bias = Parameter(b_data)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return ops.mm(input, self.weight.t()) + self.bias

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

# Need to add .t() to Tensor class
