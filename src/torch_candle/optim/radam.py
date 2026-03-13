"""torch_candle.optim.radam — RAdam optimizer"""
import numpy as np, math
from .optimizer import Optimizer

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, decoupled_weight_decay=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.rho_inf = 2.0 / (1 - betas[1]) - 1

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
                    state.update({'step': 0, 'exp_avg': np.zeros_like(grad), 'exp_avg_sq': np.zeros_like(grad)})
                state['step'] += 1; t = state['step']
                state['exp_avg'] = b1 * state['exp_avg'] + (1 - b1) * grad
                state['exp_avg_sq'] = b2 * state['exp_avg_sq'] + (1 - b2) * grad ** 2
                bias_corr1 = 1 - b1 ** t; bias_corr2 = 1 - b2 ** t
                rho = self.rho_inf - 2 * t * b2 ** t / bias_corr2
                if rho > 4:
                    rect = math.sqrt((rho - 4) * (rho - 2) * self.rho_inf /
                                     ((self.rho_inf - 4) * (self.rho_inf - 2) * rho))
                    adaptive_lr = rect / (np.sqrt(state['exp_avg_sq'] / bias_corr2) + eps)
                else:
                    adaptive_lr = 1.0
                step = lr * state['exp_avg'] / bias_corr1 * adaptive_lr
                p._tensor = Tensor((p.numpy() - step).astype(np.float32))._tensor
        return loss
