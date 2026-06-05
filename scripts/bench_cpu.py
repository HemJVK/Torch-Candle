"""
Quick CPU benchmark: torch-candle vs PyTorch
"""
import time, sys, numpy as np
import torch
import torch_candle as tc
from torch_candle.nn import functional as F_tc
import torch.nn.functional as F_pt

SHAPE = (1024, 1024)
WARMUP = 3
RUNS = 10

import statistics

def bench(fn, runs=30, warmup=5):
    for _ in range(warmup): fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

x_np = np.random.randn(*SHAPE).astype(np.float32)
x_tc = tc.Tensor(x_np.copy())
x_pt = torch.from_numpy(x_np.copy())

def speedup(tc_ms, pt_ms):
    s = pt_ms / tc_ms
    mark = "✓ FASTER" if s >= 1.0 else "✗"
    return f"{tc_ms:.3f}ms vs {pt_ms:.3f}ms  speedup={s:.2f}x {mark}"

print(f"{'Op':<22} {'Result'}")
print("-" * 70)

for name, fn_tc, fn_pt in [
    ("sigmoid",   lambda: x_tc.sigmoid(),                lambda: x_pt.sigmoid()),
    ("tanh",      lambda: x_tc.tanh(),                   lambda: x_pt.tanh()),
    ("relu",      lambda: x_tc.relu(),                   lambda: x_pt.relu()),
    ("exp",       lambda: x_tc.exp(),                    lambda: x_pt.exp()),
    ("log",       lambda: x_tc.log(),                    lambda: x_pt.log()),
    ("sqrt",      lambda: x_tc.sqrt(),                   lambda: x_pt.sqrt()),
    ("sum_all",   lambda: x_tc.sum(),                    lambda: x_pt.sum()),
    ("mean_all",  lambda: x_tc.mean(),                   lambda: x_pt.mean()),
    ("softmax",   lambda: F_tc.softmax(x_tc, dim=-1),    lambda: F_pt.softmax(x_pt, dim=-1)),
    ("gelu",      lambda: F_tc.gelu(x_tc),               lambda: F_pt.gelu(x_pt)),
    ("silu",      lambda: F_tc.silu(x_tc),               lambda: F_pt.silu(x_pt)),
    ("layer_norm",lambda: F_tc.layer_norm(x_tc, SHAPE[1]),lambda: F_pt.layer_norm(x_pt, (SHAPE[1],))),
]:
    try:
        tc_ms = bench(fn_tc)
        pt_ms = bench(fn_pt)
        print(f"  {name:<20} {speedup(tc_ms, pt_ms)}")
    except Exception as e:
        print(f"  {name:<20} ERROR: {e}")

# CrossEntropy
print()
N, C = 512, 1000
logits_np = np.random.randn(N, C).astype(np.float32)
targets_np = np.random.randint(0, C, N).astype(np.int64)
logits_tc = tc.Tensor(logits_np)
targets_tc = tc.Tensor(targets_np.astype(np.float32))
logits_pt = torch.from_numpy(logits_np)
targets_pt = torch.from_numpy(targets_np)
tc_ms = bench(lambda: F_tc.cross_entropy(logits_tc, targets_tc))
pt_ms = bench(lambda: F_pt.cross_entropy(logits_pt, targets_pt))
print(f"  {'cross_entropy':<20} {speedup(tc_ms, pt_ms)}")

# Matmul
A_tc = tc.Tensor(np.random.randn(512, 512).astype(np.float32))
B_tc = tc.Tensor(np.random.randn(512, 512).astype(np.float32))
A_pt = torch.randn(512, 512)
B_pt = torch.randn(512, 512)
tc_ms = bench(lambda: A_tc.matmul(B_tc))
pt_ms = bench(lambda: A_pt @ B_pt)
print(f"  {'matmul 512x512':<20} {speedup(tc_ms, pt_ms)}")
