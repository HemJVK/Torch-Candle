"""torch_candle.nn.modules.sparse — Sparse layers"""
import numpy as np
from ..module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init
from .. import functional as F

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
                 sparse=False, _weight=None, _freeze=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        if _weight is None:
            self.weight = Parameter(Tensor(np.empty((num_embeddings, embedding_dim), dtype=np.float32)))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim]
            self.weight = Parameter(_weight)
            
        if _freeze:
            self.weight.requires_grad = False
            
    def reset_parameters(self):
        init.normal_(self.weight)
        if self.padding_idx is not None:
            with Tensor._no_grad():
                # We need a proper no_grad block, but for now just zero it out
                idx = self.padding_idx
                arr = self.weight.numpy().copy()
                arr[idx] = 0.0
                self.weight._tensor = Tensor(arr)._tensor

    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
