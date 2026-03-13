"""torch_candle.utils.data — Dataset, DataLoader, Samplers, and utilities."""
import math
import random
import numpy as np

from .dataset import Dataset, TensorDataset
from .dataloader import DataLoader


class ConcatDataset(Dataset):
    """Concatenation of multiple datasets."""
    def __init__(self, datasets):
        self.datasets = list(datasets)
    def __len__(self): return sum(len(d) for d in self.datasets)
    def __getitem__(self, idx):
        for d in self.datasets:
            if idx < len(d): return d[idx]
            idx -= len(d)
        raise IndexError("index out of range")


class Subset(Dataset):
    """Subset of a dataset at specified indices."""
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]


def random_split(dataset, lengths, generator=None):
    """Randomly split a dataset into given lengths."""
    from ...tensor import Tensor
    total = len(dataset)
    assert sum(lengths) == total, "Lengths must sum to dataset length"
    indices = list(range(total))
    random.shuffle(indices)
    subsets = []
    offset = 0
    for l in lengths:
        subsets.append(Subset(dataset, indices[offset:offset + l]))
        offset += l
    return subsets


class Sampler:
    def __init__(self, data_source=None): self.data_source = data_source
    def __iter__(self): raise NotImplementedError
    def __len__(self): raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
    def __iter__(self): return iter(range(len(self.data_source)))
    def __len__(self): return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
    @property
    def num_samples(self): return self._num_samples or len(self.data_source)
    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(random.choices(range(n), k=self.num_samples))
        return iter(random.sample(range(n), self.num_samples))
    def __len__(self): return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch; batch = []
        if batch and not self.drop_last:
            yield batch
    def __len__(self):
        n = len(self.sampler)
        if self.drop_last: return n // self.batch_size
        return math.ceil(n / self.batch_size)


class IterableDataset(Dataset):
    def __iter__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError("IterableDataset does not support indexing")
    def __len__(self): raise TypeError("IterableDataset has no length")


def default_collate(batch):
    """Default collate function — stacks tensors, converts scalars."""
    from ...tensor import Tensor
    from ... import ops
    elem = batch[0]
    if isinstance(elem, Tensor):
        return ops.stack(batch, dim=0)
    elif isinstance(elem, (int, float)):
        return Tensor(np.array(batch, dtype=np.float32))
    elif isinstance(elem, np.ndarray):
        return Tensor(np.stack(batch, axis=0).astype(np.float32))
    elif isinstance(elem, tuple):
        return tuple(default_collate(samples) for samples in zip(*batch))
    elif isinstance(elem, dict):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, list):
        return [default_collate(samples) for samples in zip(*batch)]
    else:
        return batch
