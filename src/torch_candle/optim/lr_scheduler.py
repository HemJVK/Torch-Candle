"""torch_candle.optim.lr_scheduler — Learning rate schedulers matching PyTorch."""
import math

class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lrs[:]
        self.step()

    def get_lr(self): raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = values[:]

    def get_last_lr(self): return self._last_lr

    def state_dict(self):
        return {'last_epoch': self.last_epoch, 'base_lrs': self.base_lrs}

    def load_state_dict(self, d):
        self.last_epoch = d['last_epoch']
        self.base_lrs = d['base_lrs']


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, verbose=False):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]


class CosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        self.T_0 = T_0; self.T_mult = T_mult; self.eta_min = eta_min
        self.T_cur = 0; self.T_i = T_0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]


class ReduceLROnPlateau:
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        self.optimizer = optimizer; self.mode = mode; self.factor = factor
        self.patience = patience; self.threshold = threshold
        self.threshold_mode = threshold_mode; self.cooldown = cooldown
        self.min_lr = min_lr; self.eps = eps; self.verbose = verbose
        self.cooldown_counter = 0; self.num_bad_epochs = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics):
        current = float(metrics)
        if self.mode == 'min':
            is_better = current < self.best - self.threshold
        else:
            is_better = current > self.best + self.threshold
        if is_better:
            self.best = current; self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1; self.num_bad_epochs = 0
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_last_lr(self): return self._last_lr
    def state_dict(self): return {'best': self.best, 'num_bad_epochs': self.num_bad_epochs}
    def load_state_dict(self, d): self.best = d['best']; self.num_bad_epochs = d['num_bad_epochs']


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None,
                 mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle',
                 cycle_momentum=True, base_momentum=0.8, max_momentum=0.9,
                 last_epoch=-1, verbose=False):
        self.base_lr_input = base_lr; self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down if step_size_down else step_size_up
        self.mode = mode; self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)
        self.base_lrs = [base_lr] * len(optimizer.param_groups)

    def get_lr(self):
        cycle = math.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        scale = max(0, 1 - x)
        return [base + (self.max_lr - base) * scale for base in self.base_lrs]


class OneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None,
                 pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                 base_momentum=0.85, max_momentum=0.95, div_factor=25., final_div_factor=1e4,
                 three_phase=False, last_epoch=-1, verbose=False):
        self.max_lr = max_lr; self.total_steps = total_steps or (epochs * steps_per_epoch)
        self.pct_start = pct_start; self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch, verbose)
        self.base_lrs = [max_lr / div_factor] * len(optimizer.param_groups)

    def get_lr(self):
        t = self.last_epoch; total = self.total_steps
        peak = int(total * self.pct_start)
        if t <= peak:
            scale = t / peak
        else:
            scale = 1 - (t - peak) / (total - peak)
        scale = max(0, scale)
        return [base + (self.max_lr - base) * scale for base in self.base_lrs]


class LambdaLR(_LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.lr_lambda = [lr_lambda] * len(optimizer.param_groups) if callable(lr_lambda) else lr_lambda
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base * lmbda(self.last_epoch) for base, lmbda in zip(self.base_lrs, self.lr_lambda)]


class MultiplicativeLR(_LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.lr_lambda = [lr_lambda] * len(optimizer.param_groups) if callable(lr_lambda) else lr_lambda
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0: return self.base_lrs
        return [group['lr'] * lmbda(self.last_epoch)
                for group, lmbda in zip(self.optimizer.param_groups, self.lr_lambda)]


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, start_factor=1.0/3, end_factor=1.0, total_iters=5, last_epoch=-1, verbose=False):
        self.start_factor = start_factor; self.end_factor = end_factor; self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0: return [base * self.start_factor for base in self.base_lrs]
        if self.last_epoch > self.total_iters: return [group['lr'] for group in self.optimizer.param_groups]
        factor = (self.end_factor - self.start_factor) / (self.total_iters * self.start_factor +
                  (self.last_epoch - 1) * (self.end_factor - self.start_factor))
        return [group['lr'] * (1 + factor) for group in self.optimizer.param_groups]


class SequentialLR:
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        self.schedulers = schedulers; self.milestones = milestones
        self.optimizer = optimizer; self.last_epoch = last_epoch

    def step(self):
        self.last_epoch += 1
        idx = 0
        for i, m in enumerate(self.milestones):
            if self.last_epoch >= m: idx = i + 1
        self.schedulers[min(idx, len(self.schedulers) - 1)].step()

    def get_last_lr(self): return self.schedulers[-1].get_last_lr()


class ChainedScheduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for s in self.schedulers: s.step()

    def get_last_lr(self): return self.schedulers[-1].get_last_lr()
