import math
import random
from ...tensor import Tensor

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            # Simple collation: list of tuples -> tuple of batched tensors
            if isinstance(batch_data[0], tuple):
                collated = []
                for j in range(len(batch_data[0])):
                    tensors = [item[j] for item in batch_data]
                    # Ensure items are Tensors
                    from ... import ops
                    tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
                    collated.append(ops.stack(tensors, dim=0))
                yield tuple(collated)
            else:
                from ... import ops
                # Ensure items are Tensors
                batch_data = [t if isinstance(t, Tensor) else Tensor(t) for t in batch_data]
                yield ops.stack(batch_data, dim=0)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
