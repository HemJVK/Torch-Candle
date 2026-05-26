import time
import torch_candle as torch
from torch_candle.utils import data

# Generate a large dataset with synthetic tensor data
class LargeDataset(data.Dataset):
    def __init__(self, size=5000):
        self.size = size
        # Pre-generate tensors
        self.tensors = [torch.arange(100) * float(i) for i in range(size)]
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Simulate some minor data preparation latency
        time.sleep(0.0001)
        return self.tensors[idx]

def run_benchmark():
    dataset = LargeDataset(2000)
    
    # 1. Single Process Benchmark
    loader_single = data.DataLoader(dataset, batch_size=64, num_workers=0, shuffle=True)
    t0 = time.time()
    for batch in loader_single:
        _ = batch[0].sum().item()
    t_single = time.time() - t0
    print(f"Single-Process (num_workers=0) time: {t_single:.4f} seconds")
    
    # 2. Multi-Process Benchmark (4 Workers)
    loader_multi = data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)
    t0 = time.time()
    for batch in loader_multi:
        _ = batch[0].sum().item()
    t_multi = time.time() - t0
    print(f"Multi-Process (num_workers=4) time: {t_multi:.4f} seconds")
    
    speedup = t_single / t_multi
    print(f"Throughput Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
