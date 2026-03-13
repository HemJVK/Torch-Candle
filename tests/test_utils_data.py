import pytest
import numpy as np
import torch_candle as torch
from torch_candle.utils import data

def test_tensor_dataset():
    x = torch.arange(10)
    y = torch.tensor([i * 2.0 for i in range(10)])
    dataset = data.TensorDataset(x, y)
    
    assert len(dataset) == 10
    item = dataset[5]
    assert item[0].item() == 5
    assert item[1].item() == 10.0

def test_dataloader():
    x = torch.arange(12)
    y = torch.arange(12) * 2
    dataset = data.TensorDataset(x, y)
    loader = data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 12 / 4 = 3 batches
    assert len(loader) == 3
    batches = list(loader)
    
    b1_x, b1_y = batches[0]
    assert tuple(b1_x.shape) == (4,)
    np.testing.assert_array_equal(b1_x.numpy(), [0, 1, 2, 3])
    np.testing.assert_array_equal(b1_y.numpy(), [0, 2, 4, 6])

def test_dataloader_shuffle():
    x = torch.arange(100)
    dataset = data.TensorDataset(x)
    loader = data.DataLoader(dataset, batch_size=10, shuffle=True)
    
    batches = list(loader)
    b1_x = batches[0][0]  # unwrapping the single tensor from output tuple
    # Since it's shuffled, it shouldn't just be 0..9
    np_b1_x = b1_x.numpy()
    assert not np.array_equal(np_b1_x, np.arange(10))
    # Elements should still sum properly across all batches
    total_sum_actual = sum(b[0].sum().item() for b in batches)
    total_sum_expected = sum(range(100))
    assert total_sum_actual == total_sum_expected

def test_concat_dataset():
    d1 = data.TensorDataset(torch.arange(5))
    d2 = data.TensorDataset(torch.arange(5, 10))
    concat_d = data.ConcatDataset([d1, d2])
    
    assert len(concat_d) == 10
    assert concat_d[0][0].item() == 0
    assert concat_d[5][0].item() == 5
    assert concat_d[9][0].item() == 9
    with pytest.raises(IndexError):
        _ = concat_d[10]

def test_subset_dataset():
    d = data.TensorDataset(torch.arange(10))
    subset = data.Subset(d, [0, 2, 4, 6, 8])
    assert len(subset) == 5
    assert subset[1][0].item() == 2

def test_random_split():
    d = data.TensorDataset(torch.arange(10))
    s1, s2 = data.random_split(d, [7, 3])
    assert len(s1) == 7
    assert len(s2) == 3
    
    # check intersection is empty
    elems1 = set(s1[i][0].item() for i in range(7))
    elems2 = set(s2[i][0].item() for i in range(3))
    assert len(elems1.intersection(elems2)) == 0
