"""torch_candle.optim.rmsprop"""
import numpy as np
from .optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; alpha = group['alpha']; eps = group['eps']
            wd = group['weight_decay']; momentum = group['momentum']; centered = group['centered']
            for p in group['params']:
                if p.grad is None: continue
                from ..tensor import Tensor
                grad = p.grad.numpy() if isinstance(p.grad, Tensor) else np.array(p.grad)
                if wd != 0:
                    grad = grad + wd * p.numpy()
                state = self.state[id(p)]
                if 'square_avg' not in state:
                    state['square_avg'] = np.zeros_like(grad)
                    if momentum > 0: state['momentum_buffer'] = np.zeros_like(grad)
                    if centered: state['grad_avg'] = np.zeros_like(grad)
                state['square_avg'] = alpha * state['square_avg'] + (1 - alpha) * grad ** 2
                if centered:
                    state['grad_avg'] = alpha * state['grad_avg'] + (1 - alpha) * grad
                    avg = state['square_avg'] - state['grad_avg'] ** 2
                else:
                    avg = state['square_avg']
                if momentum > 0:
                    state['momentum_buffer'] = momentum * state['momentum_buffer'] + grad / (np.sqrt(avg) + eps)
                    update = state['momentum_buffer']
                else:
                    update = grad / (np.sqrt(avg) + eps)
                p_np = p.numpy() - lr * update
                p._tensor = Tensor(p_np.astype(np.float32))._tensor
        return loss
