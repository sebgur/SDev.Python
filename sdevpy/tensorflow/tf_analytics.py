""" Various functions in Tensorflow.
    Development has stopped. Planning to move to PyTorch.
"""
import tensorflow as tf
from sdevpy.montecarlo.smoothers import SMOOTH_STDEV


def tf_approx_cdf(x): # pragma: no cov
    """ Simple approximation of CDF """
    return 1.0 / (1.0 + tf.math.exp(-x / 0.5879))


def tf_smooth_max_diff(spot, strike): # pragma: no cov
    """ Smoothing of call payoff """
    d1 = tf.math.log(spot / strike) / SMOOTH_STDEV + 0.5 * SMOOTH_STDEV
    d2 = d1 - SMOOTH_STDEV
    n1 = tf_approx_cdf(d1)
    n2 = tf_approx_cdf(d2)
    return spot * n1 - 0.5 * strike * (n1 + n2) # Average
