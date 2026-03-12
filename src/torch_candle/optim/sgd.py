from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad._tensor
                
                if weight_decay != 0:
                    d_p = d_p + p._tensor * weight_decay

                # Basic SGD update: p = p - lr * d_p
                # In candle, we might need to replace the underlying tensor
                # because some operations are not in-place
                p._tensor = p._tensor - d_p * lr

        return loss
