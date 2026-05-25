from .optimizer import Optimizer
import numpy as np

class Rprop(Optimizer):
    """
    Rprop (Resilient Backpropagation) optimizer.
    """
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50.0)):
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)
        super(Rprop, self).__init__(params, defaults)
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            eta_minus, eta_plus = group['etas']
            step_min, step_max = group['step_sizes']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if id(p) not in self.state:
                    from .. import ones_like, zeros_like
                    self.state[id(p)] = {
                        'prev': zeros_like(p),
                        'step_size': ones_like(p) * lr
                    }
                
                state = self.state[id(p)]
                prev = state['prev']
                step_size = state['step_size']
                
                p_np = p.numpy()
                g_np = grad.numpy()
                prev_np = prev.numpy()
                ss_np = step_size.numpy()
                
                # Sign product
                prod = g_np * prev_np
                ss_np = np.where(prod > 0, np.minimum(ss_np * eta_plus, step_max), ss_np)
                ss_np = np.where(prod < 0, np.maximum(ss_np * eta_minus, step_min), ss_np)
                g_np = np.where(prod < 0, 0.0, g_np)
                
                # Update parameters
                p_np = p_np - np.sign(g_np) * ss_np
                
                from ..tensor import Tensor
                from .. import no_grad
                with no_grad():
                    p._tensor = Tensor(p_np)._tensor
                state['prev'] = Tensor(g_np)
                state['step_size'] = Tensor(ss_np)

        return loss
