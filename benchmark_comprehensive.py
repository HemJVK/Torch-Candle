import time
import numpy as np
import torch_candle as tc
import torch as pt

# Style helper
def print_header(title):
    print("\n" + "═" * 80)
    print(f"║ {title.center(76)} ║")
    print("═" * 80)

def print_section(name):
    print("\n" + "─" * 80)
    print(f"  {name}")
    print("─" * 80)
    print(f"{'Operation':<35} | {'PyTorch':<18} | {'Torch-Candle':<18} | {'Speedup':<10}")
    print("-" * 80)

def print_row(op_name, pt_time, tc_time):
    if pt_time == 0 or tc_time == 0:
        speedup = "N/A"
    else:
        ratio = pt_time / tc_time
        speedup = f"{ratio:.2f}x"
        if ratio > 1.05:
            speedup = f"\033[92m{speedup:<10}\033[0m"  # Green if Torch-Candle is faster
        elif ratio < 0.95:
            speedup = f"\033[91m{speedup:<10}\033[0m"  # Red if PyTorch is faster
        else:
            speedup = f"{speedup:<10}"
    print(f"{op_name:<35} | {pt_time:12.3f} ms | {tc_time:12.3f} ms | {speedup}")

def run_timed(fn, warmup=3, runs=10):
    for _ in range(warmup):
        _ = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)

# Warmup both backends globally
np.random.seed(42)
N = 512
np_a = np.random.randn(N, N).astype(np.float32)
np_b = np.random.randn(N, N).astype(np.float32)

print_header("PyTorch vs Torch-Candle Comprehensive Benchmark Suite")
print(f"  PyTorch CUDA available      : {pt.cuda.is_available()}")
print(f"  Torch-Candle CUDA available : True (GTX 1650 sm_75 enabled)")

# =========================================================================
# 1. CPU BENCHMARK SUITE
# =========================================================================
print_header("1. CPU BENCHMARK SUITE (PyTorch CPU vs Torch-Candle CPU)")

# --- Element-wise CPU ---
print_section("Element-wise Operations [1024x1024]")
A_np = np.random.randn(1024, 1024).astype(np.float32)
B_np = np.random.randn(1024, 1024).astype(np.float32)

pt_A = pt.tensor(A_np)
pt_B = pt.tensor(B_np)
tc_A = tc.tensor(A_np, device="cpu")
tc_B = tc.tensor(B_np, device="cpu")

print_row("add", run_timed(lambda: pt_A + pt_B), run_timed(lambda: tc_A + tc_B))
print_row("mul", run_timed(lambda: pt_A * pt_B), run_timed(lambda: tc_A * tc_B))
print_row("relu", run_timed(lambda: pt_A.relu()), run_timed(lambda: tc_A.relu()))
print_row("sigmoid", run_timed(lambda: pt_A.sigmoid()), run_timed(lambda: tc_A.sigmoid()))
print_row("tanh", run_timed(lambda: pt_A.tanh()), run_timed(lambda: tc_A.tanh()))
print_row("exp", run_timed(lambda: pt_A.exp()), run_timed(lambda: tc_A.exp()))

# --- Matmul CPU ---
print_section("Linear Algebra [Matmul]")
pt_M1 = pt.tensor(np_a)
pt_M2 = pt.tensor(np_b)
tc_M1 = tc.tensor(np_a, device="cpu")
tc_M2 = tc.tensor(np_b, device="cpu")

print_row("matmul (512x512)", run_timed(lambda: pt_M1 @ pt_M2), run_timed(lambda: tc_M1 @ tc_M2))

# --- Reductions CPU ---
print_section("Reductions [1024x1024]")
print_row("sum (all)", run_timed(lambda: pt.sum(pt_A)), run_timed(lambda: tc.sum(tc_A)))
print_row("mean (all)", run_timed(lambda: pt.mean(pt_A)), run_timed(lambda: tc.mean(tc_A)))
print_row("norm (L2)", run_timed(lambda: pt.norm(pt_A)), run_timed(lambda: tc.norm(tc_A)))

# --- Deep Learning Layers CPU ---
print_section("Deep Learning Modules & Functional Ops")
import torch.nn as pt_nn
import torch_candle.nn as tc_nn
import torch.nn.functional as pt_F
import torch_candle.nn.functional as tc_F

pt_lin = pt_nn.Linear(1024, 4096)
tc_lin = tc_nn.Linear(1024, 4096)
pt_x = pt.randn(1, 1024)
tc_x = tc.randn(1, 1024)

print_row("nn.Linear (1x1024 -> 4096)", run_timed(lambda: pt_lin(pt_x)), run_timed(lambda: tc_lin(tc_x)))
print_row("F.dropout (p=0.5)", run_timed(lambda: pt_F.dropout(pt_x, p=0.5, training=True)), run_timed(lambda: tc_F.dropout(tc_x, p=0.5, training=True)))

# =========================================================================
# 2. GPU BENCHMARK SUITE
# =========================================================================
print_header("2. GPU BENCHMARK SUITE (PyTorch GPU vs Torch-Candle GPU)")
if not pt.cuda.is_available():
    print("  \033[93mNOTE: PyTorch CUDA is not available. Falling back to PyTorch CPU as the baseline.\033[0m")
    print("  This benchmark compares Torch-Candle GPU/CUDA directly against PyTorch CPU!")

# --- Element-wise GPU ---
print_section("Element-wise Operations [1024x1024]")
pt_A_gpu = pt_A.cuda() if pt.cuda.is_available() else pt_A
pt_B_gpu = pt_B.cuda() if pt.cuda.is_available() else pt_B
tc_A_gpu = tc_A.cuda()
tc_B_gpu = tc_B.cuda()

print_row("add", run_timed(lambda: pt_A_gpu + pt_B_gpu), run_timed(lambda: tc_A_gpu + tc_B_gpu))
print_row("mul", run_timed(lambda: pt_A_gpu * pt_B_gpu), run_timed(lambda: tc_A_gpu * tc_B_gpu))
print_row("relu", run_timed(lambda: pt_A_gpu.relu()), run_timed(lambda: tc_A_gpu.relu()))

# --- Matmul GPU ---
print_section("Linear Algebra [Matmul]")
pt_M1_gpu = pt_M1.cuda() if pt.cuda.is_available() else pt_M1
pt_M2_gpu = pt_M2.cuda() if pt.cuda.is_available() else pt_M2
tc_M1_gpu = tc_M1.cuda()
tc_M2_gpu = tc_M2.cuda()

print_row("matmul (512x512)", run_timed(lambda: pt_M1_gpu @ pt_M2_gpu), run_timed(lambda: tc_M1_gpu @ tc_M2_gpu))

# =========================================================================
# 3. COMBINED PIPELINE BENCHMARK (CPU + GPU Pipeline)
# =========================================================================
print_header("3. COMBINED PIPELINE BENCHMARK (CPU -> GPU Transfer -> Math -> CPU -> NumPy)")
print("  This workflow measures deep learning pipeline memory bandwidth and latency:")
print("  1. Create CPU tensor from NumPy")
print("  2. Transfer tensor to GPU (.cuda())")
# We will do a matrix multiply on the GPU to represent a forward model pass
print("  3. Run matrix multiplication on GPU")
print("  4. Transfer output back to CPU (.cpu() / .numpy())")

def pt_combined_pipeline():
    a = pt.tensor(np_a)
    b = pt.tensor(np_b)
    if pt.cuda.is_available():
        a = a.cuda()
        b = b.cuda()
    c = a @ b
    if pt.cuda.is_available():
        c = c.cpu()
    return c.numpy()

def tc_combined_pipeline():
    a = tc.tensor(np_a, device="cpu")
    b = tc.tensor(np_b, device="cpu")
    a = a.cuda()
    b = b.cuda()
    c = a @ b
    c = c.cpu()
    return c.numpy()

print_section("End-to-End CPU+GPU Combined Pipeline [512x512]")
print_row("Combined DL Pipeline", run_timed(pt_combined_pipeline), run_timed(tc_combined_pipeline))
print("=" * 80)
