"""torch_candle.optim.nadam — NAdam optimizer"""
import numpy as np, math
from .optimizer import Optimizer

class NAdam(Optimizer):
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum_decay=4e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; b1, b2 = group['betas']; eps = group['eps']
            wd = group['weight_decay']; mu_decay = group['momentum_decay']
            for p in group['params']:
                if p.grad is None: continue
                from ..tensor import Tensor
                grad = p.grad.numpy() if isinstance(p.grad, Tensor) else np.array(p.grad)
                if wd != 0: grad = grad + wd * p.numpy()
                state = self.state[id(p)]
                if 'step' not in state:
                    state.update({'step': 0, 'exp_avg': np.zeros_like(grad), 'exp_avg_sq': np.zeros_like(grad)})
                state['step'] += 1; t = state['step']
                mu = b1 * (1 - 0.5 * 0.96 ** (t * mu_decay))
                mu_next = b1 * (1 - 0.5 * 0.96 ** ((t + 1) * mu_decay))
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * grad
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * grad ** 2
                denom = np.sqrt(state['exp_avg_sq'] / (1 - b2 ** t)) + eps
                step = lr * (mu_next * state['exp_avg'] / (1 - b1 ** (t + 1)) +
                             (1 - mu) * grad / (1 - b1 ** t)) / denom
                p._tensor = Tensor((p.numpy() - step).astype(np.float32))._tensor
        return loss
