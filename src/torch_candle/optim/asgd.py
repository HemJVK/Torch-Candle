from .optimizer import Optimizer

class ASGD(Optimizer):
    """
    Averaged Stochastic Gradient Descent (ASGD) optimizer.
    """
    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)
        super(ASGD, self).__init__(params, defaults)
        self.state = {}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            lambd = group['lambd']
            alpha = group['alpha']
            t0 = group['t0']
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
                        'step': 0,
                        'eta': lr,
                        'mu': 1.0,
                        'ax': zeros_like(p)
                    }
                
                state = self.state[id(p)]
                state['step'] += 1
                t = state['step']
                
                # ASGD update:
                # p = p - eta * grad
                from .. import no_grad
                with no_grad():
                    p._tensor = (p - grad * state['eta'])._tensor
                
                # eta = lr / (1 + lambd * lr * t)^alpha
                state['eta'] = lr / ((1.0 + lambd * lr * t) ** alpha)
                
                # ax = ax + mu * (p - ax)
                mu = 1.0 / max(1.0, t - t0)
                state['mu'] = mu
                state['ax'] = state['ax'] + (p - state['ax']) * mu

        return loss
