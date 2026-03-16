from .tensor import Tensor

def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None):
    """
    Computes the reverse-mode auto-differentiation gradients.
    Delegates to the Rust backend directly for extreme performance over PyO3.
    """
    if grad_variables is not None:
        grad_tensors = grad_variables
        
    if isinstance(tensors, Tensor):
        tensors = (tensors,)
    else:
        tensors = tuple(tensors)
        
    if grad_tensors is None:
        grad_tensors = (None,) * len(tensors)
    elif isinstance(grad_tensors, Tensor):
        grad_tensors = (grad_tensors,)
    else:
        grad_tensors = tuple(grad_tensors)
        
    for t, g in zip(tensors, grad_tensors):
        t.backward(g)

class Function:
    """
    Base class to create custom autograd functions.
    In torch_candle, the native graph handles most operations, 
    but this stub exists for PyTorch API compatibility.
    """
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError
        
    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError

class SavedTensor:
    """Stub for saved tensor."""
    pass

def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False):
    """
    Computes and returns the sum of gradients of outputs with respect to the inputs.
    """
    backward(outputs, grad_tensors=grad_outputs, retain_graph=retain_graph, create_graph=create_graph)
    
    if isinstance(inputs, Tensor):
        return (inputs.grad,)
    return tuple(i.grad for i in inputs)

class set_grad_enabled:
    def __init__(self, mode: bool):
        self.prev = Tensor._grad_enabled
        self.mode = mode
    def __enter__(self):
        Tensor._grad_enabled = self.mode
    def __exit__(self, *args):
        Tensor._grad_enabled = self.prev

class no_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(False)

class enable_grad(set_grad_enabled):
    def __init__(self):
        super().__init__(True)
