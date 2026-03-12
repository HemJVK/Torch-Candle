class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = defaults
        # params can be an iterator of Parameters
        self.param_groups = []
        param_group = {'params': list(params)}
        param_group.update(defaults)
        self.param_groups.append(param_group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad._tensor = p.grad._tensor.zeros_like()

    def step(self, closure=None):
        raise NotImplementedError
