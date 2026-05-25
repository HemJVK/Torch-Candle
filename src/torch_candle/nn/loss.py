from .module import Module
from .. import ops
import numpy as np

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction

class MSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        diff = input - target
        loss = diff * diff
        if self.reduction == 'mean':
            num_elements = input._tensor.nelements if hasattr(input._tensor, 'nelements') else np.prod(input.shape)
            return loss.sum() * (1.0 / num_elements)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', label_smoothing=0.0):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        from . import functional as F
        return F.cross_entropy(input, target, reduction=self.reduction)


class L1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        from . import functional as F
        return F.l1_loss(input, target, reduction=self.reduction)


class BCELoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.weight = weight

    def forward(self, input, target):
        from . import functional as F
        return F.binary_cross_entropy(input, target, weight=self.weight, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super().__init__(size_average, reduce, reduction)
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, input, target):
        from . import functional as F
        return F.binary_cross_entropy_with_logits(
            input, target, weight=self.weight, reduction=self.reduction, pos_weight=self.pos_weight
        )
