""" Utilities for Black-Scholes model """
import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
import tensorflow as tf
import tensorflow_probability as tfp
from sdevpy import settings

N = scipy.stats.norm.cdf
tf_N = tfp.distributions.Normal(0.0, 1.0)


def price_and_greeks(expiry, strike, spot, vol, rate, div):
    """ Calculate call PV and sensitivities by AAD on Black-Scholes CF """
    tf_spot = tf.convert_to_tensor(spot, dtype='float32')
    tf_vol = tf.convert_to_tensor(vol)
    tf_time = tf.convert_to_tensor(expiry, dtype='float32')
    tf_rate = tf.convert_to_tensor(rate, dtype='float32')
    tf_div = tf.constant(div)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([tf_spot, tf_vol, tf_time, tf_rate])
        with tf.GradientTape(persistent=True) as tape2nd:
            tape2nd.watch([tf_spot, tf_vol])

            fwd = tf_spot * tf.math.exp((tf_rate - tf_div) * tf_time)
            stdev = tf_vol * tf.math.sqrt(tf_time)
            d1 = tf.math.log(fwd / strike) / stdev + 0.5 * stdev
            d2 = d1 - stdev
            df = tf.math.exp(-tf_rate * tf_time)
            pv = df * (fwd * tf_N.cdf(d1) - strike * tf_N.cdf(d2))

        # Calculate delta and vega
        g_delta = tape2nd.gradient(pv, tf_spot)
        g_vega = tape2nd.gradient(pv, tf_vol)

    delta = tape.gradient(pv, tf_spot)
    gamma = tape.gradient(g_delta, tf_spot)
    vega = tape.gradient(pv, tf_vol)
    theta = tape.gradient(pv, tf_time)
    dv01 = tape.gradient(pv, tf_rate)
    volga = tape.gradient(g_vega, tf_vol)
    vanna = tape.gradient(g_delta, tf_vol)

    # Scale
    vega = vega.numpy() * settings.VEGA_SCALING
    theta = -theta.numpy() * settings.THETA_SCALING
    dv01 = dv01.numpy() * settings.DV01_SCALING
    volga = volga.numpy() * settings.VOLGA_SCALING
    vanna = vanna.numpy() * settings.VANNA_SCALING

    return [pv.numpy(), delta.numpy(), gamma.numpy(), vega, theta, dv01, volga, vanna]
