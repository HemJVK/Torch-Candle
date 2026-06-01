import faulthandler
faulthandler.enable()

import torch_candle as torch
from torch_candle.func import jacrev, hessian

f = lambda x: x * x
x = torch.Tensor([3.0])

print("1. Calling jacrev...")
j_val = jacrev(f)(x)
print("jacrev result:", j_val.numpy())

print("2. Calling hessian...")
h_val = hessian(f)(x)
print("hessian result:", h_val.numpy())
