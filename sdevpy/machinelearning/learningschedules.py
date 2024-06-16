""" Custom learning schedules """
import tensorflow as tf
import math
from sdevpy.tools.constants import TWO_PI

# Custom learning rate scheduler, exponentially decreases between given values
class FlooredExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Custom learning rate scheduler, exponentially decreases between given values """
    def __init__(self, num_samples, batch_size, target_epoch, initial_lr=1e-1, final_lr=1e-4):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        # self.decay = decay
        # self.decay_steps = decay_steps
        # A step is the usage of one gradient, i.e. for one batch. As we go through the whole sample
        # in 1 epoch, the number of steps per epoch is given by the number of batches per epoch
        # i.e. the formula below.
        steps_per_epoch = num_samples / batch_size
        percent_reached = 0.10  # Percentage of the final LR reached by the chosen epoch
        self.decay = final_lr * percent_reached / (initial_lr - final_lr)
        self.steps_to_target = np.float32(steps_per_epoch * target_epoch)

    def __call__(self, step):
        ratio = tf.cast(step / self.steps_to_target, tf.float32)
        coeff = tf.pow(self.decay, ratio)
        ampl = self.initial_lr - self.final_lr
        return self.final_lr + ampl * coeff

    # def __call__(self, step):
    #     ratio = tf.cast(step / self.decay_steps, tf.float32)
    #     coeff = tf.pow(self.decay, ratio)
    #     return self.initial_lr * coeff + self.final_lr * (1.0 - coeff)

    def get_config(self):
        config = { 'initial_lr': self.initial_lr,
                   'final_lr': self.final_lr,
                   'decay': self.decay,
                   'decay_steps': self.steps_to_target }
        return config

import numpy as np

# Custom learning rate scheduler, cyclically exponentially decreases between given values
class CyclicalExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Custom learning rate scheduler, cyclically exponentially decreases between given values """
    def __init__(self, num_samples, batch_size, target_epoch, initial_lr=1e-1, final_lr=1e-4,
                 periods=10.0):
        # Amplitude decay
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        # self.period = periods
        # A step is the usage of one gradient, i.e. for one batch. As we go through the whole sample
        # in 1 epoch, the number of steps per epoch is given by the number of batches per epoch
        # i.e. the formula below.
        steps_per_epoch = num_samples / batch_size
        percent_reached = 0.10  # Percentage of the final LR reached by the chosen epoch
        self.decay = final_lr * percent_reached / (initial_lr - final_lr)
        self.steps_to_target = np.float32(steps_per_epoch * target_epoch)

        # Oscillations
        self.steps_per_period = np.float32(target_epoch * steps_per_epoch / periods)

    def __call__(self, step):
        ratio = tf.cast(step / self.steps_to_target, tf.float32)
        coeff = tf.pow(self.decay, ratio)
        ampl = self.initial_lr - self.final_lr
        two_pi = tf.cast(TWO_PI, tf.float32)
        arg = tf.cast(step / self.steps_per_period, tf.float32)
        oscillation = (2.0 + tf.math.cos(arg * two_pi)) / 2.0  # Between 0.5 and 1.5
        ampl = ampl * oscillation
        return self.final_lr + ampl * coeff

    def get_config(self):
        config = { 'initial_lr': self.initial_lr,
                   'final_lr': self.final_lr,
                   'decay': self.decay,
                   'steps_to_target': self.steps_to_target,
                   'steps_per_period': self.steps_per_period }
        return config
