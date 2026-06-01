import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(workspace_root, "src"))

import torch_candle as torch
from torch_candle import Tensor

x = Tensor([3.0])
diff = Tensor([1.0], device=x.device)
print("1. calling to_grad_tensor on x._tensor...")
xtg = x._tensor.to_grad_tensor(diff._tensor)
print("2. wrapping xtg as Tensor...")
x_grad = Tensor(xtg)
print("3. executing x_grad * x_grad...")
res = x_grad * x_grad
print("4. done.")
