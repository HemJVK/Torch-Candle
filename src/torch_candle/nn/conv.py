"""torch_candle.nn.conv — Convolutional layers backed by Candle Rust via PyO3."""
from .module import Module
from .parameter import Parameter
from . import functional as F
import numpy as np
import math

try:
    import candle as _candle
except ImportError:
    _candle = None


class _ConvNd(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, bias, padding_mode):
        super().__init__()
        from ..tensor import Tensor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        k = 1.0 / (groups * in_channels * math.prod(kernel_size))
        k = math.sqrt(k)

        weight_shape = (out_channels, in_channels // groups, *kernel_size)
        w_data = np.random.uniform(-k, k, weight_shape).astype(np.float32)
        self.weight = Parameter(Tensor(w_data))

        if bias:
            b_data = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = Parameter(Tensor(b_data))
        else:
            self.register_parameter('bias', None)


class Conv1d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        if isinstance(kernel_size, int): kernel_size = (kernel_size,)
        if isinstance(stride, int): stride = (stride,)
        if isinstance(padding, int): padding = (padding,)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)

    def forward(self, input):
        # Treat as 2D conv with height=1
        from ..tensor import Tensor
        import numpy as np
        x = input.numpy()
        # x: (N, C, L) -> (N, C, 1, L)
        x4d = x[:, :, np.newaxis, :]
        from ..tensor import Tensor as T
        inp4d = T(x4d)
        out4d = F.conv2d(inp4d, self.weight, self.bias, self.stride[0], self.padding[0],
                          self.dilation, self.groups)
        # out4d: (N, C_out, 1, L_out) -> (N, C_out, L_out)
        return T(out4d.numpy()[:, :, 0, :])

    def __repr__(self):
        return (f"Conv1d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride})")


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class ConvTranspose2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)
        self.output_padding = output_padding

    def forward(self, input):
        # Simplified transpose conv via numpy
        from ..tensor import Tensor
        import numpy as np
        x = input.numpy()
        N, C_in, H_in, W_in = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        pH, pW = self.padding
        H_out = (H_in - 1) * sH - 2 * pH + kH
        W_out = (W_in - 1) * sW - 2 * pW + kW
        C_out = self.out_channels
        out = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
        w = self.weight.numpy()  # (C_in, C_out, kH, kW)
        for n in range(N):
            for c_in in range(C_in):
                for c_out in range(C_out):
                    for i in range(H_in):
                        for j in range(W_in):
                            h_start = i * sH - pH
                            w_start = j * sW - pW
                            out[n, c_out, max(0,h_start):h_start+kH, max(0,w_start):w_start+kW] += (
                                x[n, c_in, i, j] * w[c_in, c_out,
                                    max(0,-h_start):kH, max(0,-w_start):kW])
        if self.bias is not None:
            out += self.bias.numpy().reshape(1, C_out, 1, 1)
        return Tensor(out)
