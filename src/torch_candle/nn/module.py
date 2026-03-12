from collections import OrderedDict
from .parameter import Parameter

class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self, recurse=True):
        for name, param in self._parameters.items():
            yield param
        if recurse:
            for name, module in self._modules.items():
                yield from module.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        for name, param in self._parameters.items():
            yield prefix + name, param
        if recurse:
            for name, module in self._modules.items():
                submodule_prefix = prefix + name + '.'
                yield from module.named_parameters(submodule_prefix, recurse)

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(*input)

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        for param in self.parameters():
            param._tensor = param.to(*args, **kwargs)._tensor
        for module in self._modules.values():
            module.to(*args, **kwargs)
        return self
