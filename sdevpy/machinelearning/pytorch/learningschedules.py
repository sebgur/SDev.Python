""" Custom learning rate schedules for PyTorch.
    Uses the decorator-based registry design, so new subclasses automatically register provided
    they are decorated.
"""
import math
from torch.optim.lr_scheduler import (LRScheduler, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    OneCycleLR, LambdaLR)


_SCHEDULER_REGISTRY = {}


def register_scheduler(name: str):
    """ Register scheduler class corresponding to name string """
    def decorator(cls):
        """ This is the decorator that gets added at compile time by register.
            It is then called when the class object (not an instance) is built by the compiler.
            So it is when the class object is built by the compiler that the name gets registered.
        """
        _SCHEDULER_REGISTRY[name] = cls
        return cls
    return decorator


def create_scheduler(name: str, optimizer, **kwargs):
    """ Create a learning rate scheduler given its name, the optimizer, and parameters """
    try:
        cls = _SCHEDULER_REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {sorted(_SCHEDULER_REGISTRY)}") from e

    return cls(optimizer, **kwargs)


# Register built-in schedulers (note that LambdaLR is multiplicative by definition)
built_ins = {"step": StepLR, "multistep": MultiStepLR, "exponential": ExponentialLR,
             "cosine": CosineAnnealingLR, "onecycle": OneCycleLR, "lambda": LambdaLR}
for _name, _cls in built_ins.items():
    register_scheduler(_name)(_cls)


@register_scheduler("constant")
class ConstantLR(LRScheduler):
    """ The learning rate is constant, as set by init_lr in the optimizer """
    def __init__(self, optimizer, last_epoch: int=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return list(self.base_lrs)


@register_scheduler("warmup_cosine")
class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        return [
            self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


@register_scheduler("floored_exponentialdecay")
class FlooredExponentialDecay(LRScheduler):
    """ Exponentially decays LR between initial_lr and final_lr over target_epoch epochs """
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


@register_scheduler("cyclical_exponentialdecay")
class CyclicalExponentialDecay(LRScheduler):
    """ Exponentially decays LR amplitude with a cosine oscillation over each period """
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


if __name__ == "__main__":
    print("Hello")
    optimizer = None
    scheduler = create_scheduler("warmup_cosine", optimizer, warmup_epochs=5, total_epochs=50)
