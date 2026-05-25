"""
benchmark.py — Comprehensive PyTorch vs torch-candle performance comparison.

Run with:
    PYTHONPATH=src python3 benchmark.py
"""

import sys
import time
import math
import statistics
from typing import Callable

import numpy as np

# ── Library imports ────────────────────────────────────────────────────────────
try:
    import torch as _torch
    _PYTORCH_OK = True
except ImportError:
    _PYTORCH_OK = False
    print("[WARNING] PyTorch not installed — skipping PyTorch timings.")

try:
    sys.path.insert(0, "src")
    import torch_candle as _candle
    import torch_candle.nn as _candle_nn
    import torch_candle.nn.functional as _candle_F
    _CANDLE_OK = True
except ImportError as e:
    _CANDLE_OK = False
    print(f"[WARNING] torch_candle not available — skipping Candle timings. ({e})")

# ── Benchmark Helpers ──────────────────────────────────────────────────────────

WARMUP = 3
RUNS   = 10

def timeit(fn: Callable, warmup=WARMUP, runs=RUNS) -> float:
    """Run fn() `warmup` times, then `runs` times. Returns median ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)

WIDTH = 72
COL   = 20

def hdr(title: str):
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)
    _print_row("Operation", "PyTorch (ms)", "Candle (ms)", "Speedup")
    print("-" * WIDTH)

def _print_row(name, pt, can, ratio):
    pt_s  = f"{pt:>10.3f}" if isinstance(pt, float) else f"{pt:>{COL}}"
    can_s = f"{can:>10.3f}" if isinstance(can, float) else f"{can:>{COL}}"
    rat_s = f"{ratio:>8.2f}x" if isinstance(ratio, float) else f"{ratio:>{COL}}"
    print(f"  {name:<30} {pt_s}  {can_s}  {rat_s}")

def row(name, pt_fn, can_fn):
    try:
        pt_ms  = timeit(pt_fn)  if pt_fn  else None
    except Exception as e:
        pt_ms = None
        print(f"  [WARN] PyTorch  {name}: {e}")
    try:
        can_ms = timeit(can_fn) if can_fn else None
    except Exception as e:
        can_ms = None
        print(f"  [WARN] Candle   {name}: {e}")
    if pt_ms and can_ms:
        ratio = pt_ms / can_ms
    else:
        ratio = None
    _print_row(
        name,
        pt_ms  if pt_ms  else "N/A",
        can_ms if can_ms else "N/A",
        ratio  if ratio  else "N/A",
    )


# ── 1. Element-wise Ops ────────────────────────────────────────────────────────

def bench_elementwise():
    hdr("1. Element-wise Operations  [shape: 1024×1024]")
    N = 1024

    if _PYTORCH_OK:
        pt_a = _torch.randn(N, N)
        pt_b = _torch.randn(N, N)
    if _CANDLE_OK:
        ca_a = _candle.randn(N, N)
        ca_b = _candle.randn(N, N)

    row("add",
        (lambda: (pt_a + pt_b))                 if _PYTORCH_OK else None,
        (lambda: (ca_a + ca_b))                 if _CANDLE_OK  else None)
    row("mul",
        (lambda: (pt_a * pt_b))                 if _PYTORCH_OK else None,
        (lambda: (ca_a * ca_b))                 if _CANDLE_OK  else None)
    row("relu",
        (lambda: _torch.relu(pt_a))             if _PYTORCH_OK else None,
        (lambda: _candle.relu(ca_a))            if _CANDLE_OK  else None)
    row("sigmoid",
        (lambda: _torch.sigmoid(pt_a))          if _PYTORCH_OK else None,
        (lambda: ca_a.sigmoid())                if _CANDLE_OK  else None)
    row("tanh",
        (lambda: _torch.tanh(pt_a))             if _PYTORCH_OK else None,
        (lambda: ca_a.tanh())                   if _CANDLE_OK  else None)
    row("exp",
        (lambda: _torch.exp(pt_a))              if _PYTORCH_OK else None,
        (lambda: _candle.exp(ca_a))             if _CANDLE_OK  else None)
    row("log",
        (lambda: _torch.log(pt_a.abs() + 1e-5)) if _PYTORCH_OK else None,
        (lambda: _candle.log(ca_a.abs() + 1e-5)) if _CANDLE_OK else None)
    row("sqrt",
        (lambda: _torch.sqrt(pt_a.abs()))       if _PYTORCH_OK else None,
        (lambda: _candle.sqrt(ca_a.abs() + 1e-5)) if _CANDLE_OK else None)


# ── 2. Linear Algebra ──────────────────────────────────────────────────────────

def bench_linalg():
    hdr("2. Linear Algebra  [matmul: 512×512 @ 512×512]")
    N = 512

    if _PYTORCH_OK:
        pt_a = _torch.randn(N, N)
        pt_b = _torch.randn(N, N)
    if _CANDLE_OK:
        ca_a = _candle.randn(N, N)
        ca_b = _candle.randn(N, N)

    row("matmul (512×512)",
        (lambda: _torch.mm(pt_a, pt_b))   if _PYTORCH_OK else None,
        (lambda: _candle.mm(ca_a, ca_b))  if _CANDLE_OK  else None)

    if _PYTORCH_OK:
        pt_a2 = _torch.randn(128, 128)
        pt_b2 = _torch.randn(128, 128)
    if _CANDLE_OK:
        ca_a2 = _candle.randn(128, 128)
        ca_b2 = _candle.randn(128, 128)

    row("matmul (128×128)",
        (lambda: _torch.mm(pt_a2, pt_b2)) if _PYTORCH_OK else None,
        (lambda: _candle.mm(ca_a2, ca_b2)) if _CANDLE_OK else None)

    if _PYTORCH_OK:
        pt_a3 = _torch.randn(4, 128, 128)
        pt_b3 = _torch.randn(4, 128, 128)
    if _CANDLE_OK:
        ca_a3 = _candle.randn(4, 128, 128)
        ca_b3 = _candle.randn(4, 128, 128)

    row("bmm (4×128×128)",
        (lambda: _torch.bmm(pt_a3, pt_b3)) if _PYTORCH_OK else None,
        (lambda: _candle.bmm(ca_a3, ca_b3)) if _CANDLE_OK else None)

    if _PYTORCH_OK:
        pt_v = _torch.randn(N)
        pt_w = _torch.randn(N, N)
    if _CANDLE_OK:
        ca_v = _candle.randn(N)
        ca_w = _candle.randn(N, N)

    row("mv (x @ W, 512)",
        (lambda: _torch.mv(pt_w, pt_v))   if _PYTORCH_OK else None,
        (lambda: _candle.mv(ca_w, ca_v))  if _CANDLE_OK  else None)


# ── 3. Reductions ─────────────────────────────────────────────────────────────

def bench_reductions():
    hdr("3. Reduction Ops  [shape: 1024×1024]")
    N = 1024

    if _PYTORCH_OK:
        pt_a = _torch.randn(N, N)
    if _CANDLE_OK:
        ca_a = _candle.randn(N, N)

    row("sum (all)",
        (lambda: pt_a.sum())              if _PYTORCH_OK else None,
        (lambda: ca_a.sum())              if _CANDLE_OK  else None)
    row("sum (dim=1)",
        (lambda: pt_a.sum(dim=1))        if _PYTORCH_OK else None,
        (lambda: ca_a.sum(dim=1))        if _CANDLE_OK  else None)
    row("mean (all)",
        (lambda: pt_a.mean())            if _PYTORCH_OK else None,
        (lambda: ca_a.mean())            if _CANDLE_OK  else None)
    row("max (all)",
        (lambda: pt_a.max())             if _PYTORCH_OK else None,
        (lambda: ca_a.max())             if _CANDLE_OK  else None)
    row("min (all)",
        (lambda: pt_a.min())             if _PYTORCH_OK else None,
        (lambda: ca_a.min())             if _CANDLE_OK  else None)
    row("std",
        (lambda: pt_a.std())             if _PYTORCH_OK else None,
        (lambda: ca_a.std())             if _CANDLE_OK  else None)
    row("norm (L2)",
        (lambda: _torch.norm(pt_a))      if _PYTORCH_OK else None,
        (lambda: _candle.norm(ca_a))     if _CANDLE_OK  else None)


# ── 4. Neural Network Functional ──────────────────────────────────────────────

def bench_nn_functional():
    hdr("4. nn.Functional Ops  [batch=64  seq/C=256  D=512]")
    B, C, S = 64, 256, 512

    if _PYTORCH_OK:
        import torch.nn.functional as _torch_F
        pt_x  = _torch.randn(B, C)
        pt_s  = _torch.randn(B, S, C)
        pt_ln = _torch.randn(B, C)
    if _CANDLE_OK:
        ca_x  = _candle.randn(B, C)
        ca_s  = _candle.randn(B, S, C)
        ca_ln = _candle.randn(B, C)

    row("softmax (dim=1)",
        (lambda: _torch_F.softmax(pt_x, dim=1))        if _PYTORCH_OK else None,
        (lambda: _candle_F.softmax(ca_x, dim=1))        if _CANDLE_OK  else None)
    row("log_softmax (dim=1)",
        (lambda: _torch_F.log_softmax(pt_x, dim=1))    if _PYTORCH_OK else None,
        (lambda: _candle_F.log_softmax(ca_x, dim=1))   if _CANDLE_OK  else None)
    row("layer_norm (B×C)",
        (lambda: _torch_F.layer_norm(pt_ln, [C]))       if _PYTORCH_OK else None,
        (lambda: _candle_F.layer_norm(ca_ln, [C]))      if _CANDLE_OK  else None)
    row("gelu",
        (lambda: _torch_F.gelu(pt_x))                  if _PYTORCH_OK else None,
        (lambda: _candle_F.gelu(ca_x))                 if _CANDLE_OK  else None)
    row("silu",
        (lambda: _torch_F.silu(pt_x))                  if _PYTORCH_OK else None,
        (lambda: _candle_F.silu(ca_x))                 if _CANDLE_OK  else None)
    row("dropout (p=0.5)",
        (lambda: _torch_F.dropout(pt_x, 0.5, training=True)) if _PYTORCH_OK else None,
        (lambda: _candle_F.dropout(ca_x, 0.5, training=True)) if _CANDLE_OK else None)


# ── 5. nn.Linear (Forward Pass) ───────────────────────────────────────────────

def bench_linear():
    hdr("5. nn.Linear Forward Pass")

    configs = [(64, 512, 512), (64, 512, 2048), (1, 1024, 4096)]
    for B, I, O in configs:
        if _PYTORCH_OK:
            pt_lin = _torch.nn.Linear(I, O)
            pt_x   = _torch.randn(B, I)
        if _CANDLE_OK:
            ca_lin = _candle_nn.Linear(I, O)
            ca_x   = _candle.randn(B, I)

        row(f"Linear {B}×{I} → {O}",
            (lambda: pt_lin(pt_x))  if _PYTORCH_OK else None,
            (lambda: ca_lin(ca_x))  if _CANDLE_OK  else None)


# ── 6. Conv2d Forward Pass ────────────────────────────────────────────────────

def bench_conv():
    hdr("6. Conv2d Forward Pass")

    configs = [
        (1,  1,  28, 28,  8, 3),   # MNIST-like
        (4,  3, 224, 224, 16, 3),   # ImageNet-like (small batch)
    ]
    for N, Ci, H, W, Co, k in configs:
        if _PYTORCH_OK:
            pt_m = _torch.nn.Conv2d(Ci, Co, k, padding=0)
            pt_x = _torch.randn(N, Ci, H, W)
        if _CANDLE_OK:
            ca_m = _candle_nn.Conv2d(Ci, Co, k, padding=0)
            ca_x = _candle.randn(N, Ci, H, W)

        label = f"Conv2d {N}×{Ci}×{H}×{W} k={k} out={Co}"
        row(label,
            (lambda: pt_m(pt_x))  if _PYTORCH_OK else None,
            (lambda: ca_m(ca_x))  if _CANDLE_OK  else None)


# ── 7. MLP Inference ──────────────────────────────────────────────────────────

def bench_mlp_inference():
    hdr("7. MLP Inference  [64 × (512→512→512→10)]")
    B = 64

    if _PYTORCH_OK:
        pt_model = _torch.nn.Sequential(
            _torch.nn.Linear(512, 512), _torch.nn.ReLU(),
            _torch.nn.Linear(512, 512), _torch.nn.ReLU(),
            _torch.nn.Linear(512, 10),
        )
        pt_x = _torch.randn(B, 512)
        with _torch.no_grad():
            pass  # warmup compile

    if _CANDLE_OK:
        class MLP(_candle_nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = _candle_nn.Linear(512, 512)
                self.fc2 = _candle_nn.Linear(512, 512)
                self.fc3 = _candle_nn.Linear(512, 10)

            def forward(self, x):
                x = _candle.relu(self.fc1(x))
                x = _candle.relu(self.fc2(x))
                return self.fc3(x)

        ca_model = MLP()
        ca_x = _candle.randn(B, 512)

    with _torch.no_grad() if _PYTORCH_OK else _DummyCtx():
        row("MLP forward (no grad)",
            (lambda: pt_model(pt_x))  if _PYTORCH_OK else None,
            (lambda: ca_model(ca_x))  if _CANDLE_OK  else None)

    row("MLP forward+backward",
        (lambda: _pt_train_step(pt_model, pt_x)) if _PYTORCH_OK else None,
        (lambda: _ca_train_step(ca_model, ca_x))  if _CANDLE_OK  else None)


class _DummyCtx:
    def __enter__(self): pass
    def __exit__(self, *a): pass


def _pt_train_step(model, x):
    model.zero_grad()
    y    = model(x)
    loss = y.mean()
    loss.backward()


def _ca_train_step(model, x):
    for p in model.parameters():
        p._tensor.grad = None
    y    = model(x)
    loss = y.mean()
    loss.backward()


# ── 8. Loss Functions ─────────────────────────────────────────────────────────

def bench_losses():
    hdr("8. Loss Functions  [B=256 C=1000]")
    B, C = 256, 1000

    if _PYTORCH_OK:
        pt_pred = _torch.randn(B, C)
        pt_tgt  = _torch.randint(0, C, (B,))
    if _CANDLE_OK:
        ca_pred = _candle.randn(B, C)
        ca_tgt  = _candle.randint(0, C, (B,))

    if _PYTORCH_OK:
        pt_mse_pred = _torch.randn(B, C)
        pt_mse_tgt  = _torch.randn(B, C)
    if _CANDLE_OK:
        ca_mse_pred = _candle.randn(B, C)
        ca_mse_tgt  = _candle.randn(B, C)

    row("MSELoss",
        (lambda: _torch.nn.MSELoss()(pt_mse_pred, pt_mse_tgt)) if _PYTORCH_OK else None,
        (lambda: _candle_nn.MSELoss()(ca_mse_pred, ca_mse_tgt)) if _CANDLE_OK else None)
    row("CrossEntropyLoss",
        (lambda: _torch.nn.CrossEntropyLoss()(pt_pred, pt_tgt)) if _PYTORCH_OK else None,
        (lambda: _candle_nn.CrossEntropyLoss()(ca_pred, ca_tgt)) if _CANDLE_OK else None)


# ── 9. Tensor Creation ────────────────────────────────────────────────────────

def bench_creation():
    hdr("9. Tensor Creation  [1024×1024]")
    N = 1024

    row("zeros",
        (lambda: _torch.zeros(N, N))      if _PYTORCH_OK else None,
        (lambda: _candle.zeros(N, N))     if _CANDLE_OK  else None)
    row("ones",
        (lambda: _torch.ones(N, N))       if _PYTORCH_OK else None,
        (lambda: _candle.ones(N, N))      if _CANDLE_OK  else None)
    row("randn",
        (lambda: _torch.randn(N, N))      if _PYTORCH_OK else None,
        (lambda: _candle.randn(N, N))     if _CANDLE_OK  else None)
    row("arange (0..N*N)",
        (lambda: _torch.arange(N * N).reshape(N, N))  if _PYTORCH_OK else None,
        (lambda: _candle.arange(N * N).reshape(N, N)) if _CANDLE_OK  else None)


# ── 10. Attention (Scaled Dot-Product) ───────────────────────────────────────

def bench_attention():
    hdr("10. Scaled Dot-Product Attention  [B=4 H=8 S=512 D=64]")
    B, H, S, D = 4, 8, 256, 64

    if _PYTORCH_OK:
        Q = _torch.randn(B * H, S, D)
        K = _torch.randn(B * H, S, D)
        V = _torch.randn(B * H, S, D)
    if _CANDLE_OK:
        cQ = _candle.randn(B * H, S, D)
        cK = _candle.randn(B * H, S, D)
        cV = _candle.randn(B * H, S, D)

    import torch.nn.functional as _torch_F
    row("SDPA (no mask)",
        (lambda: _torch_F.scaled_dot_product_attention(Q, K, V)) if _PYTORCH_OK else None,
        (lambda: _candle_F.scaled_dot_product_attention(cQ, cK, cV)) if _CANDLE_OK else None)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        PyTorch vs torch-candle  ─  Performance Benchmark            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    pt_ver = _torch.__version__ if _PYTORCH_OK else "N/A"
    print(f"  PyTorch  version : {pt_ver}")
    print(f"  Candle   variant : torch_candle (CPU, Rust backend)")
    print(f"  Warmup runs      : {WARMUP}   Timed runs: {RUNS}")
    print(f"  Timing metric    : median wall-clock ms")
    print()
    print("  Speedup > 1.00x  ⇒  torch-candle is faster")
    print("  Speedup < 1.00x  ⇒  PyTorch is faster")

    bench_elementwise()
    bench_linalg()
    bench_reductions()
    bench_nn_functional()
    bench_linear()
    bench_conv()
    bench_mlp_inference()
    bench_losses()
    bench_creation()
    bench_attention()

    print()
    print("=" * WIDTH)
    print("  Benchmark complete.")
    print("=" * WIDTH)
    print()


if __name__ == "__main__":
    main()
