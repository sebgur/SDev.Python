""" Utilities for Black-Scholes model """
import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
import py_vollib.black.implied_volatility as jaeckel
import tensorflow as tf
import tensorflow_probability as tfp
from sdevpy import settings

N = scipy.stats.norm.cdf
# Ninv = scipy.stats.norm.ppf
tf_N = tfp.distributions.Normal(0.0, 1.0)


def price(expiry, strike, is_call, fwd, vol):
    """ Option price under the Black-Scholes model """
    w = 1.0 if is_call else -1.0
    s = vol * np.sqrt(expiry)
    d1 = np.log(fwd / strike) / s + 0.5 * s
    d2 = d1 - s
    return w * (fwd * N(w * d1) - strike * N(w * d2))

def implied_vol_jaeckel(expiry, strike, is_call, fwd, fwd_price):
    """ Black-Scholes implied volatility using P. Jaeckel's 'Let's be rational' method,
        from package py_vollib. Install with pip install py_vollib or at
        https://pypi.org/project/py_vollib/. Unfortunately we found it has instabilities
        near ATM. """
    flag = 'c' if is_call else 'p'
    p = fwd_price
    iv = jaeckel.implied_volatility_of_undiscounted_option_price(p, fwd, strike, expiry, flag)
    return iv

def implied_vol(expiry, strike, is_call, fwd, fwd_price):
    """ Direct method by numerical inversion using Brent """
    options = {'xtol': 1e-4, 'maxiter': 100, 'disp': False}
    xmin = 1e-6
    xmax = 1.0

    def error(vol):
        premium = price(expiry, strike, is_call, fwd, vol)
        return (premium - fwd_price) ** 2

    res = minimize_scalar(fun=error, bracket=(xmin, xmax), options=options, method='brent')

    return res.x


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

# def performance(spot_vol, repo_rate, div_rate, expiry, strike, fixings):
#     shape = spot_vol.shape
#     num_underlyings = int(shape[0] / 2)
#     # print(num_underlyings)
#     forward_perf = 1.0
#     vol2 = 0.0
#     for i in range(num_underlyings):
#         forward = spot_vol[2 * i] * np.exp((repo_rate - div_rate) * expiry)
#         perf = forward  # / fixings[i]
#         # perf = forward / fixings[i]
#         forward_perf = forward_perf * perf
#         vol = spot_vol[2 * i + 1]
#         vol2 = vol2 + np.power(vol, 2)
#
#     vol = np.sqrt(vol2)
#     # print(vol)
#
#     # return forward_perf
#     return black_formula(forward_perf, strike, vol, expiry, True)

if __name__ == "__main__":
    EXPIRY = 1.0
    VOL = 0.25
    IS_CALL = True
    NUM_POINTS = 100
    # FWD = 100
    # K = 100
    # p = price(EXPIRY, K, IS_CALL, FWD, VOL)
    # iv = implied_vol(EXPIRY, K, IS_CALL, FWD, p)
    # print(iv)
    f_space = np.linspace(100, 120, NUM_POINTS)
    k_space = np.linspace(20, 2180, NUM_POINTS)
    prices = price(EXPIRY, k_space, IS_CALL, f_space, VOL)
    # print(prices)
    implied_vols = []
    for i, k in enumerate(k_space):
        implied_vols.append(implied_vol(EXPIRY, k, IS_CALL, f_space[i], prices[i]))

    # print(implied_vols)
