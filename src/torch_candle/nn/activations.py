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
