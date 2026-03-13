"""torch_candle.nn.modules.rnn — Recurrent layers"""
from ..module import Module
from ..parameter import Parameter
from ...tensor import Tensor
from .. import init
from .. import functional as F
import math

class RNNBase(Module):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0.0, bidirectional=False, proj_size=0):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        
        # simplified parameter creation
        pass

    def forward(self, input, hx=None):
        pass # To be implemented via F.rnn/lstm/gru

class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('RNN', *args, **kwargs)

class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)

class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('GRU', *args, **kwargs)
