from .optimizer import Optimizer
import math
import torch_candle_backend as _kernels

class Adam(Optimizer):
    """
    Adam optimizer: Adaptive Moment Estimation.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)
        
        self.state = {}  # Stores first and second moments

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if id(p) not in self.state:
                    from .. import zeros_like
                    self.state[id(p)] = {
                        'step': 0,
                        'exp_avg': zeros_like(p),
                        'exp_avg_sq': zeros_like(p),
                    }
                
                state = self.state[id(p)]
                state['step'] += 1
                t = state['step']
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                if weight_decay != 0:
                    grad = grad + p * weight_decay
                
                # Update biased first/second moment estimates
                state['exp_avg'] = exp_avg * beta1 + grad * (1.0 - beta1)
                state['exp_avg_sq'] = exp_avg_sq * beta2 + (grad * grad) * (1.0 - beta2)
                
                # Bias-corrected estimates
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute denominator: sqrt(exp_avg_sq) + eps
                denom = state['exp_avg_sq'].sqrt() + eps
                
                # Parameter update — ensure we don't track this updated as part of autograd
                from .. import no_grad
                with no_grad():
                    # Direct tensor update via Rust backend to avoid creating subgraph
                    update = state['exp_avg'] * (step_size / denom)
                    p._tensor = (p - update)._tensor

        return loss
