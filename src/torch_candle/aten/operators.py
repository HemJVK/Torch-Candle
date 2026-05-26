def add(self, other):
    from torch_candle import ops
    return ops.add(self, other)

def sub(self, other):
    from torch_candle import ops
    return ops.sub(self, other)

def mul(self, other):
    from torch_candle import ops
    return ops.mul(self, other)

def div(self, other):
    from torch_candle import ops
    return ops.div(self, other)

def matmul(self, other):
    from torch_candle import ops
    return ops.matmul(self, other)

def relu(self):
    from torch_candle import ops
    return ops.relu(self)

def sum(self, dim=None, keepdim=False):
    from torch_candle import ops
    return ops.sum(self, dim, keepdim)
