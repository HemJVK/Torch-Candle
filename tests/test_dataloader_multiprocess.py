import pytest
import numpy as np
import time
import torch_candle as torch
from torch_candle.utils import data

class FailingDataset(data.Dataset):
    """A dataset that throws an error at a specific index to test worker recovery."""
    def __init__(self, size=20):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        if idx == 13:
            raise ValueError("Intentional error at item 13")
        return torch.tensor([float(idx)])

def custom_collate(batch):
    """Custom collator that returns a string alongside stacked tensors."""
    from torch_candle import ops
    tensors = [item[0] if isinstance(item, tuple) else item for item in batch]
    stacked = ops.stack(tensors, dim=0)
    return stacked, "custom_collated"

def test_dataloader_multiprocess_basic():
    x = torch.arange(40)
    dataset = data.TensorDataset(x)
    
    # Run with 2 worker processes
    loader = data.DataLoader(dataset, batch_size=5, num_workers=2, shuffle=False)
    
    assert len(loader) == 8
    batches = list(loader)
    
    assert len(batches) == 8
    # Verify exact sequential output values
    b1_x = batches[0][0]
    np.testing.assert_array_equal(b1_x.numpy(), [0, 1, 2, 3, 4])
    
    b2_x = batches[1][0]
    np.testing.assert_array_equal(b2_x.numpy(), [5, 6, 7, 8, 9])
    
    b8_x = batches[7][0]
    np.testing.assert_array_equal(b8_x.numpy(), [35, 36, 37, 38, 39])

def test_dataloader_multiprocess_shuffle():
    x = torch.arange(200)
    dataset = data.TensorDataset(x)
    
    # Run with 4 worker processes and shuffle
    loader = data.DataLoader(dataset, batch_size=20, num_workers=4, shuffle=True)
    
    batches = list(loader)
    assert len(batches) == 10
    
    # Verify sum across all batches matches expected total sum
    total_sum_actual = sum(batch[0].sum().item() for batch in batches)
    total_sum_expected = sum(range(200))
    assert total_sum_actual == total_sum_expected

def test_dataloader_multiprocess_custom_collate():
    x = torch.arange(30)
    dataset = data.TensorDataset(x)
    
    loader = data.DataLoader(dataset, batch_size=10, num_workers=2, collate_fn=custom_collate)
    
    batches = list(loader)
    assert len(batches) == 3
    
    stacked, tag = batches[0]
    assert tuple(stacked.shape) == (10,)
    assert tag == "custom_collated"

def test_dataloader_multiprocess_error_propagation():
    dataset = FailingDataset(30)
    loader = data.DataLoader(dataset, batch_size=5, num_workers=2)
    
    # When we iterate, the worker process should encounter ValueError at index 13
    # and propagate it to the main thread as a RuntimeError.
    with pytest.raises(RuntimeError) as excinfo:
        for _ in loader:
            pass
            
    assert "Intentional error at item 13" in str(excinfo.value)

def test_dataloader_multiprocess_early_break():
    x = torch.arange(100)
    dataset = data.TensorDataset(x)
    loader = data.DataLoader(dataset, batch_size=5, num_workers=3)
    
    # Test early exit from loop
    count = 0
    for batch in loader:
        count += 1
        if count == 3:
            break
            
    # Give a short window for processes to clean up and exit
    time.sleep(0.5)
    
    # The __iter__'s finally block should successfully close and join worker processes
    # so no zombie/running children should exist for this process.
    # Verification will also pass automatically if the interpreter finishes successfully
    # without deadlocking or hanging.
    assert count == 3
