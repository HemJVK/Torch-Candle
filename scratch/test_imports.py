print("Before import", flush=True)
import torch_candle as torch
from torch_candle.func import jacrev, hessian
print("Creating tensor...", flush=True)
f = lambda x: x * x
x = torch.Tensor([3.0])
print("Calling jacrev...", flush=True)
j_val = jacrev(f)(x)
print(f"jacrev: {j_val.numpy()}", flush=True)
print("Calling hessian...", flush=True)
h_val = hessian(f)(x)
print(f"hessian: {h_val.numpy()}", flush=True)
