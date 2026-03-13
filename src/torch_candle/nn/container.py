from .module import Module

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self, name, module):
        setattr(self, name, module)
        return module

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(dict(list(self._modules.items())[idx]))
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)
