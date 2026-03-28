""" Payoff smoothers for numerical methods """
import numpy as np
import tensorflow as tf


# Smoothing parameters (for the max function)
SMOOTH_VOL = 0.40
SMOOTH_TIME = 10.0 / 365.0
SMOOTH_STDEV = SMOOTH_VOL * np.sqrt(SMOOTH_TIME)


# Numpy versions (non-AAD)

def approx_cdf(x):
    """ Simple approximation of CDF """
    return 1.0 / (1.0 + np.exp(-x / 0.5879))


def smooth_max_diff(spot, strike):
    """ Smoothing of call payoff """
    d1 = np.log(spot / strike) / SMOOTH_STDEV + 0.5 * SMOOTH_STDEV
    d2 = d1 - SMOOTH_STDEV
    n1 = approx_cdf(d1)
    n2 = approx_cdf(d2)
    return spot * n1 - 0.5 * strike * (n1 + n2)  # Average


# Tensorflow versions (AAD)

def tf_approx_cdf(x):
    """ Simple approximation of CDF """
    return 1.0 / (1.0 + tf.math.exp(-x / 0.5879))

def tf_smooth_max_diff(spot, strike):
    """ Smoothing of call payoff """
    d1 = tf.math.log(spot / strike) / SMOOTH_STDEV + 0.5 * SMOOTH_STDEV
    d2 = d1 - SMOOTH_STDEV
    n1 = tf_approx_cdf(d1)
    n2 = tf_approx_cdf(d2)
    return spot * n1 - 0.5 * strike * (n1 + n2) # Average
