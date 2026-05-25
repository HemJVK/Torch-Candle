from .module import Module
from .. import ops
import math

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ops.relu(input)

    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # sigmoid(x) = 1 / (1 + exp(-x))
        return 1.0 / (1.0 + (-input).exp())

    def __repr__(self):
        return "Sigmoid()"

class GELU(Module):
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, input):
        if self.approximate == 'tanh':
            return 0.5 * input * (1.0 + ops.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * input.pow(3))))
        return input * 0.5 * (1.0 + ops.erf(input / math.sqrt(2.0)))

    def __repr__(self):
        return f"GELU(approximate='{self.approximate}')"

class SiLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * (1.0 / (1.0 + (-input).exp()))

    def __repr__(self):
        return "SiLU()"

class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        dim = self.dim if self.dim is not None else -1
        # stable softmax using logsumexp
        return (input - ops.logsumexp(input, dim, keepdim=True)).exp()

    def __repr__(self):
        return f"Softmax(dim={self.dim})"


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.tanh()

    def __repr__(self):
        return "Tanh()"


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        from . import functional as F
        return F.leaky_relu(input, self.negative_slope)

    def __repr__(self):
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        from . import functional as F
        return F.elu(input, self.alpha)

    def __repr__(self):
        return f"ELU(alpha={self.alpha})"


class SELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        from . import functional as F
        return F.selu(input)

    def __repr__(self):
        return "SELU()"


class PReLU(Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.num_parameters = num_parameters
        import numpy as np
        from ..tensor import Tensor
        from .parameter import Parameter
        self.weight = Parameter(Tensor(np.full((num_parameters,), init, dtype=np.float32)))

    def forward(self, input):
        from . import functional as F
        return F.prelu(input, self.weight)

    def __repr__(self):
        return f"PReLU(num_parameters={self.num_parameters})"
