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

class Context:
    def __init__(self):
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)


class Function:
    """
    Base class to create custom autograd functions.
    Supports standard forward/backward method definitions.
    """
    _tape = []

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )
        
        detached_args = []
        for arg in args:
            if isinstance(arg, Tensor):
                detached_args.append(arg.detach())
            else:
                detached_args.append(arg)
                
        out_val = cls.forward(ctx, *detached_args, **kwargs)
        
        if not isinstance(out_val, Tensor):
            out_val = Tensor(out_val)
            
        if requires_grad and Tensor._grad_enabled:
            out_val.requires_grad = True
            # Store on the execution tape
            Function._tape.append((cls, ctx, args, out_val))
            
        return out_val


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

class standard_mode:
    """Context manager to run standard autograd without self-healing reconstruction."""
    def __init__(self):
        self.prev = True
    def __enter__(self):
        self.prev = getattr(Tensor, "enable_sha", True)
        Tensor.enable_sha = False
        return self
    def __exit__(self, *args):
        Tensor.enable_sha = self.prev

