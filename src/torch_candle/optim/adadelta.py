from .optimizer import Optimizer

class Adadelta(Optimizer):
    """
    Adadelta optimizer.
    """
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super(Adadelta, self).__init__(params, defaults)
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if weight_decay != 0:
                    grad = grad + p * weight_decay
                
                if id(p) not in self.state:
                    from .. import zeros_like
                    self.state[id(p)] = {
                        'square_avg': zeros_like(p),
                        'acc_delta': zeros_like(p)
                    }
                
                state = self.state[id(p)]
                sq_avg = state['square_avg']
                acc_delta = state['acc_delta']
                
                # Accumulate gradient square avg
                state['square_avg'] = sq_avg * rho + (grad * grad) * (1.0 - rho)
                
                # Compute update
                # std = sqrt(square_avg + eps)
                std = (state['square_avg'] + eps).sqrt()
                # delta = sqrt(acc_delta + eps) / std * grad
                delta = (acc_delta + eps).sqrt() / std * grad
                
                # Apply update
                from .. import no_grad
                with no_grad():
                    p._tensor = (p - delta * lr)._tensor
                
                # Accumulate delta square avg
                state['acc_delta'] = acc_delta * rho + (delta * delta) * (1.0 - rho)

        return loss
