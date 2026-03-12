from ..tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data=None, requires_grad=True):
        if data is None:
            super().__init__([], requires_grad=requires_grad)
        else:
            super().__init__(data, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter({super().__repr__()})"
