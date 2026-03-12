from .module import Module
from .. import ops

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
