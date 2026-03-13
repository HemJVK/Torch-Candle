"""torch_candle.optim.adamax"""
import numpy as np
from .optimizer import Optimizer

class Adamax(Optimizer):
    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; b1, b2 = group['betas']; eps = group['eps']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                from ..tensor import Tensor
                grad = p.grad.numpy() if isinstance(p.grad, Tensor) else np.array(p.grad)
                if wd != 0: grad = grad + wd * p.numpy()
                state = self.state[id(p)]
                if 'step' not in state:
                    state['step'] = 0; state['exp_avg'] = np.zeros_like(grad); state['exp_inf'] = np.zeros_like(grad)
                state['step'] += 1
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * grad
                state['exp_inf'] = np.maximum(b2 * state['exp_inf'], np.abs(grad))
                bias_correction = 1 - b1 ** state['step']
                clr = lr / bias_correction
                p_np = p.numpy() - clr * state['exp_avg'] / (state['exp_inf'] + eps)
                p._tensor = Tensor(p_np.astype(np.float32))._tensor
        return loss
