from .optimizer import Optimizer
import math

class AdamW(Optimizer):
    """
    AdamW optimizer: Decoupled Weight Decay Regularization with high-performance in-place updates.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
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
        super(AdamW, self).__init__(params, defaults)
        
        self.state = {}

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
                
                # Decoupled weight decay applied in-place!
                from .. import no_grad
                with no_grad():
                    if weight_decay != 0:
                        decay_factor = 1.0 - lr * weight_decay
                        p *= decay_factor
                
                # Update biased first moment estimates in-place
                exp_avg *= beta1
                exp_avg += grad * (1.0 - beta1)
                
                # Update biased second moment estimates in-place
                exp_avg_sq *= beta2
                exp_avg_sq += (grad * grad) * (1.0 - beta2)
                
                # Bias-corrected estimates
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute denominator: sqrt(exp_avg_sq) + eps
                denom = exp_avg_sq.sqrt()
                denom += eps
                
                with no_grad():
                    # Apply update in-place!
                    update = exp_avg * (step_size / denom)
                    p -= update

        return loss
