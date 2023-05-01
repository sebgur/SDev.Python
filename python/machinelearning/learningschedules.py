""" Custom learning schedules """
import tensorflow as tf

# Custom learning rate scheduler, exponentially decreases between given values
class FlooredExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Custom learning rate scheduler, exponentially decreases between given values """
    def __init__(self, initial_lr=1e-1, final_lr=1e-4, decay=0.96, decay_steps=100):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay = decay
        self.decay_steps = decay_steps

    def __call__(self, step):
        ratio = tf.cast(step / self.decay_steps, tf.float32)
        coeff = tf.pow(self.decay, ratio)
        return self.initial_lr * coeff + self.final_lr * (1.0 - coeff)

    def get_config(self):
        config = { 'initial_lr': self.initial_lr,
                   'final_lr': self.final_lr,
                   'decay': self.decay,
                   'decay_steps': self.decay_steps }
        return config
