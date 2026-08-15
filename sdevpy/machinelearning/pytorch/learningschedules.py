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
built_ins = {"multistep": MultiStepLR, "exponential": ExponentialLR,
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


@register_scheduler("step")
class StepDecay(StepLR):
    """ Decays by steps, with multiplicative coefficient gamma """
    def __init__(self, optimizer, n_samples: int, batch_size: int, n_epochs_between_drops: int,
                 gamma: float, last_epoch: int=-1):
        steps_per_epoch = n_samples / batch_size
        step_size = n_epochs_between_drops * steps_per_epoch
        super().__init__(optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch) # Must come last


@register_scheduler("linear")
class LinearLR(LambdaLR):
    """ Linear from init_lr to final_lr at last step """
    def __init__(self, optimizer, n_samples: int, batch_size: int, n_epochs: int,
                 final_lr: float, last_epoch: int=-1):
        steps_per_epoch = n_samples / batch_size
        total_steps = n_epochs * steps_per_epoch
        initial_lr = optimizer.param_groups[0]['lr'] # Retrieve the initial LR
        final_ratio = final_lr / initial_lr
        coeff = (1.0 - final_ratio) / total_steps
        def linear_decay(step):
            return max(final_ratio, 1.0 - coeff * step)

        super().__init__(optimizer, lr_lambda=linear_decay, last_epoch=last_epoch)


@register_scheduler("floored_exponentialdecay")
class FlooredExponentialDecay(LRScheduler):
    """ Exponentially decays LR between initial_lr and final_lr over target_epoch epochs """
    def __init__(self, optimizer, n_samples: int, batch_size: int, target_epoch: int,
                 final_lr, last_epoch: int=-1):
        self.initial_lr = optimizer.param_groups[0]['lr'] # Retrieve the initial LR
        self.final_lr = final_lr
        steps_per_epoch = n_samples / batch_size
        percent_reached = 0.10
        self.decay = self.final_lr * percent_reached / (self.initial_lr - self.final_lr)
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
    """ Exponentially decays LR amplitude with a cosine oscillation """
    def __init__(self, optimizer, n_samples: int, batch_size: int, target_epoch: int,
                 final_lr, periods: float=10.0, last_epoch: int=-1):
        self.initial_lr = optimizer.param_groups[0]['lr'] # Retrieve the initial LR
        self.final_lr = final_lr
        steps_per_epoch = n_samples / batch_size
        percent_reached = 0.10
        self.decay = final_lr * percent_reached / (self.initial_lr - self.final_lr)
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
