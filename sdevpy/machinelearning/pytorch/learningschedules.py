""" Custom learning rate schedules for PyTorch """
import math
from torch.optim.lr_scheduler import LRScheduler


class FlooredExponentialDecay(LRScheduler):
    """Exponentially decays LR between initial_lr and final_lr over target_epoch epochs."""
    def __init__(self, optimizer, num_samples: int, batch_size: int, target_epoch: int,
                 initial_lr: float=1e-1, final_lr: float=1e-4, last_epoch: int=-1):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        steps_per_epoch = num_samples / batch_size
        percent_reached = 0.10
        self.decay = final_lr * percent_reached / (initial_lr - final_lr)
        self.steps_to_target = steps_per_epoch * target_epoch
        super().__init__(optimizer, last_epoch) # Must come last

    def get_lr(self):
        step = self.last_epoch
        ratio = step / self.steps_to_target
        coeff = self.decay ** ratio
        lr = self.final_lr + (self.initial_lr - self.final_lr) * coeff
        return [lr for _ in self.optimizer.param_groups]


class CyclicalExponentialDecay(LRScheduler):
    """Exponentially decays LR amplitude with a cosine oscillation over each period."""
    def __init__(self, optimizer, num_samples: int, batch_size: int, target_epoch: int,
                 initial_lr: float=1e-1, final_lr: float=1e-4, periods: float=10.0, last_epoch: int=-1):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        steps_per_epoch = num_samples / batch_size
        percent_reached = 0.10
        self.decay = final_lr * percent_reached / (initial_lr - final_lr)
        self.steps_to_target = steps_per_epoch * target_epoch
        self.steps_per_period = target_epoch * steps_per_epoch / periods
        super().__init__(optimizer, last_epoch) # Must come last

    def get_lr(self):
        step = self.last_epoch
        ratio = step / self.steps_to_target
        coeff = self.decay ** ratio
        oscillation = (2.0 + math.cos(math.tau * step / self.steps_per_period)) / 2.0
        ampl = (self.initial_lr - self.final_lr) * oscillation
        lr = self.final_lr + ampl * coeff
        return [lr for _ in self.optimizer.param_groups]
