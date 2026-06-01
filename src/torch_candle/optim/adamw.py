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

        import torch_candle_backend as _kernels

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
                
                # Retrieve the underlying numpy arrays for parameter, grad, exp_avg, and exp_avg_sq
                p_arr = p.numpy()
                g_arr = grad.numpy()
                m_arr = exp_avg.numpy()
                v_arr = exp_avg_sq.numpy()
                
                # Perform the SIMD-accelerated zero-allocation updates directly in Rust
                _kernels.fast_adamw_step(
                    p_arr,
                    g_arr,
                    m_arr,
                    v_arr,
                    beta1,
                    beta2,
                    lr,
                    weight_decay,
                    eps,
                    t
                )
                
                # Update the underlying PyTensor handles with the mutated values
                from .. import no_grad
                with no_grad():
                    p._tensor = _kernels.PyTensor(p_arr, device=p.device)
                    exp_avg._tensor = _kernels.PyTensor(m_arr, device=p.device)
                    exp_avg_sq._tensor = _kernels.PyTensor(v_arr, device=p.device)

        return loss
