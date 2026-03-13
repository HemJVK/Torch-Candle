from .optimizer import Optimizer
import math

try:
    import candle
except ImportError:
    candle = None


class Adam(Optimizer):
    """
    Adam optimizer: Adaptive Moment Estimation.
    
    Args:
        params: iterable of parameters to optimize
        lr (float): learning rate (default: 1e-3)
        betas (Tuple[float, float]): coefficients for computing running averages
        eps (float): term added for numerical stability
        weight_decay (float): L2 penalty (default: 0)
        amsgrad (bool): use AMSGrad variant (default: False)
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
                
                # Keep grad as torch_candle.Tensor for consistent arithmetic
                grad = p.grad
                
                # Initialise state using zeros_like from torch_candle
                if id(p) not in self.state:
                    from .. import zeros
                    self.state[id(p)] = {
                        'step': 0,
                        'exp_avg': zeros(p.shape, device=p.device, dtype=p.dtype),
                        'exp_avg_sq': zeros(p.shape, device=p.device, dtype=p.dtype),
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
                
                # Parameter update
                p._tensor = (p - state['exp_avg'] * step_size / denom)._tensor

        return loss
