from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer (with optional momentum).
    
    Args:
        params: iterable of parameters to optimize
        lr (float): learning rate (required)
        momentum (float): momentum factor (default: 0)
        weight_decay (float): L2 penalty (default: 0)
        nesterov (bool): enables Nesterov momentum (default: False)
    """
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue

                # Use torch_candle.Tensor directly for consistent behavior
                d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p + p * weight_decay

                if momentum != 0:
                    if id(p) not in self.state:
                        from .. import zeros
                        self.state[id(p)] = {
                            'momentum_buffer': zeros(p.shape, device=p.device, dtype=p.dtype)
                        }
                    buf = self.state[id(p)]['momentum_buffer']
                    # buf = buf * momentum + d_p * (1 - dampening)
                    buf = buf * momentum + d_p * (1.0 - dampening)
                    self.state[id(p)]['momentum_buffer'] = buf
                    
                    if nesterov:
                        d_p = d_p + buf * momentum
                    else:
                        d_p = buf

                # p = p - lr * d_p
                p._tensor = (p - d_p * lr)._tensor

        return loss
