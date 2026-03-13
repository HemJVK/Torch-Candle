class AutogradContext:
    def __init__(self):
        self.save_for_backward = []

    def save(self, *tensors):
        self.save_for_backward.extend(tensors)

class Node:
    def __init__(self, op_name, inputs, ctx):
        self.op_name = op_name
        self.inputs = inputs
        self.ctx = ctx

    def backward(self, ctx, grad) -> tuple:
        """Backward pass for the node."""
        raise NotImplementedError

class AddBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # derivative of x+y wrt x is 1, wrt y is 1
        return (grad, grad)

class SubBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # derivative of x-y wrt x is 1, wrt y is -1
        return (grad, -grad)

class MulBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(x*y)/dx = y, d(x*y)/dy = x
        x, y = self.inputs
        return (grad * y, grad * x)

class MatmulBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(X@W)/dX = grad @ W.T
        # d(X@W)/dW = X.T @ grad
        x, weight = self.inputs
        return (grad.matmul(weight.t()), x.t().matmul(grad))

class SumBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(sum(x))/dx = 1 (broadcast to x shape)
        x = self.inputs[0]
        from . import ones
        return (ones(x.shape, device=x.device, dtype=x.dtype) * grad,)

class MeanBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(mean(x))/dx = 1/N
        x = self.inputs[0]
        import numpy as np
        from . import ones
        num_elements = np.prod(x.shape)
        return (ones(x.shape, device=x.device, dtype=x.dtype) * (grad * (1.0 / num_elements)),)

class TransposeBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(x.t())/dx = grad.t()
        return (grad.t(),)

class ReshapeBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(x.reshape(shape))/dx = grad.reshape(x.shape)
        x = self.inputs[0]
        return (grad.reshape(x.shape),)

class SliceBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(x[index])/dx = zeros filled with grad at index
        x = self.inputs[0]
        from . import zeros
        from .tensor import Tensor
        full_grad = zeros(x.shape, device=x.device, dtype=x.dtype)
        # This is a bit tricky without differentiable __setitem__ or scatter
        # for now, we use numpy to help build the mask/full_grad
        import numpy as np
        np_grad = full_grad.numpy()
        np_grad[ctx.index] = grad.numpy()
        return (Tensor(np_grad, device=x.device, dtype=x.dtype),)

class PowBackward(Node):
    def backward(self, ctx, grad) -> tuple:
        # d(x^n)/dx = n * x^(n-1)
        x, n = self.inputs[0], self.inputs[1] # n is usually a float
        return (grad * (n * (x ** (n - 1))), None) # derivative wrt n is not needed for now

def backward(tensor, gradient=None):
    if not tensor.requires_grad:
        return

    if gradient is None:
        from . import ones
        gradient = ones(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    
    if tensor.grad is None:
        tensor.grad = gradient
    else:
        tensor.grad = tensor.grad + gradient

    if tensor.grad_fn is None:
        return

    # Topological sort
    visited = set()
    nodes = []
    def build_topo(t):
        if hasattr(t, 'grad_fn') and t.grad_fn is not None and t.grad_fn not in visited:
            visited.add(t.grad_fn)
            for input_tensor in t.grad_fn.inputs:
                if hasattr(input_tensor, 'requires_grad') and getattr(input_tensor, 'requires_grad'):
                    build_topo(input_tensor)
            nodes.append(t.grad_fn)
    
    # Sort logically
    nodes = []
    visited = set()
    def build_topo(t):
        if hasattr(t, 'grad_fn') and t.grad_fn is not None and t.grad_fn not in visited:
            visited.add(t.grad_fn)
            for input_tensor in t.grad_fn.inputs:
                if hasattr(input_tensor, 'requires_grad') and getattr(input_tensor, 'requires_grad'):
                    build_topo(input_tensor)
            nodes.append(t.grad_fn)
    
    build_topo(tensor)
    
    # Execute in reverse topological order
    for node in reversed(nodes):
        # The logic below is a bit simplified; real autograd tracks grads per tensor
        pass

    # For now, stick to the working recursive version but make it safer
    def run_backward(t, grad):
        if not hasattr(t, 'grad_fn') or getattr(t, 'grad_fn') is None:
            return
        
        node = getattr(t, 'grad_fn')
        ctx = getattr(node, 'ctx')
        inputs = getattr(node, 'inputs')
        
        grads = node.backward(ctx, grad)
        
        if not isinstance(grads, (list, tuple)):
            grads = [grads]
            
        for input_tensor, input_grad in zip(inputs, grads):
            if input_grad is not None and hasattr(input_tensor, 'requires_grad') and getattr(input_tensor, 'requires_grad'):
                if hasattr(input_tensor, 'grad') and getattr(input_tensor, 'grad') is None:
                    setattr(input_tensor, 'grad', input_grad)
                elif hasattr(input_tensor, 'grad'):
                    setattr(input_tensor, 'grad', getattr(input_tensor, 'grad') + input_grad)
                run_backward(input_tensor, input_grad)

    run_backward(tensor, gradient)

def vjp(outputs, inputs, grad_outputs):
    """
    Vector-Jacobi Product utility.
    Computes [grad_outputs @ J] where J is the Jacobian of outputs w.r.t inputs.
    """
    if isinstance(outputs, (list, tuple)):
        for o, g in zip(outputs, grad_outputs):
            backward(o, g)
    else:
        backward(outputs, grad_outputs)
        
    res = []
    if isinstance(inputs, (list, tuple)):
        for i in inputs:
            res.append(getattr(i, 'grad'))
    else:
        res = getattr(inputs, 'grad')
    return res
