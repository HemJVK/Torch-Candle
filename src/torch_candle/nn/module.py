from collections import OrderedDict
from .parameter import Parameter

class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def register_parameter(self, name, param):
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot register {type(param)} as parameter")
        else:
            self._parameters[name] = param
        super().__setattr__(name, param)

    def register_buffer(self, name, tensor, persistent=True):
        self._buffers[name] = tensor
        super().__setattr__(name, tensor)

    def parameters(self, recurse=True):
        for name, param in self._parameters.items():
            if param is not None:
                yield param
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        for name, param in self._parameters.items():
            if param is not None:
                yield prefix + name, param
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + name + '.'
                    yield from module.named_parameters(submodule_prefix, recurse)

    def buffers(self, recurse=True):
        for name, buf in self._buffers.items():
            if buf is not None:
                yield buf
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.buffers(recurse)

    def named_buffers(self, prefix='', recurse=True):
        for name, buf in self._buffers.items():
            if buf is not None:
                yield prefix + name, buf
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + name + '.'
                    yield from module.named_buffers(submodule_prefix, recurse)

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        for param in self.parameters():
            if param is not None:
                param._tensor = param.to(*args, **kwargs)._tensor
        for name, buf in self._buffers.items():
            if buf is not None:
                buf._tensor = buf.to(*args, **kwargs)._tensor
        for module in self._modules.values():
            if module is not None:
                module.to(*args, **kwargs)
        return self

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars=keep_vars)
        return destination

    def load_state_dict(self, state_dict, strict=True):
        for name, param in self._parameters.items():
            full_name = name
            if full_name in state_dict:
                input_param = state_dict[full_name]
                if isinstance(input_param, Parameter):
                    param._tensor = input_param._tensor
                else:
                    from ..tensor import Tensor
                    param._tensor = Tensor(input_param)._tensor
        for name, buf in self._buffers.items():
            full_name = name
            if full_name in state_dict:
                input_buf = state_dict[full_name]
                from ..tensor import Tensor
                if isinstance(input_buf, Tensor):
                    buf._tensor = input_buf._tensor
                else:
                    buf._tensor = Tensor(input_buf)._tensor
        for name, module in self._modules.items():
            if module is not None:
                sub_state = {k[len(name) + 1:]: v for k, v in state_dict.items() if k.startswith(name + '.')}
                module.load_state_dict(sub_state, strict=strict)

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        self._modules[name] = module
        super().__setattr__(name, module)
