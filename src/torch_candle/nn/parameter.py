from ..tensor import Tensor

class Parameter(Tensor):
    def __init__(self, data=None):
        if data is None:
            super().__init__([], requires_grad=True)
        elif isinstance(data, Tensor):
            super().__init__(data._tensor, requires_grad=True)
        else:
            super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter({super().__repr__()})"
