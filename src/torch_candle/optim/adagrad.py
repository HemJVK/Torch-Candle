"""torch_candle.optim.adagrad"""
import numpy as np
from .optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0,
                 initial_accumulator_value=0, eps=1e-10):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[id(p)]
                state['step'] = 0
                state['sum'] = np.full(p.shape, initial_accumulator_value, dtype=np.float32)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']; eps = group['eps']
            wd = group['weight_decay']; lr_decay = group['lr_decay']
            for p in group['params']:
                if p.grad is None: continue
                from ..tensor import Tensor
                grad = p.grad.numpy() if isinstance(p.grad, Tensor) else np.array(p.grad)
                state = self.state[id(p)]
                state['step'] += 1
                if wd != 0:
                    grad = grad + wd * p.numpy()
                effective_lr = lr / (1 + (state['step'] - 1) * lr_decay)
                state['sum'] += grad ** 2
                std = np.sqrt(state['sum']) + eps
                p_np = p.numpy() - effective_lr * grad / std
                p._tensor = Tensor(p_np.astype(np.float32))._tensor
        return loss
