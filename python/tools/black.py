import scipy.stats
import numpy as np
from numpy import sqrt, log, exp, pi

N = scipy.stats.norm.cdf
Ninv = scipy.stats.norm.ppf


def black_formula(f, k, vol_, t, is_call_):
    w = 1.0 if is_call_ else -1.0
    s = vol_ * sqrt(t)
    d1 = log(f / k) / s + 0.5 * s
    d2 = d1 - s
    return w * (f * N(w * d1) - k * N(w * d2))


def black_performance(spot_vol, repo_rate, div_rate, expiry, strike, fixings):
    shape = spot_vol.shape
    num_underlyings = int(shape[0] / 2)
    # print(num_underlyings)
    forward_perf = 1.0
    vol2 = 0.0
    for i in range(num_underlyings):
        forward = spot_vol[2 * i] * np.exp((repo_rate - div_rate) * expiry)
        perf = forward  # / fixings[i]
        # perf = forward / fixings[i]
        forward_perf = forward_perf * perf
        vol = spot_vol[2 * i + 1]
        vol2 = vol2 + np.power(vol, 2)

    vol = np.sqrt(vol2)
    # print(vol)

    # return forward_perf
    return black_formula(forward_perf, strike, vol, expiry, True)

# T = 1.0
# vol = 0.25
# F = 100
# is_call = True
#
# t_space = T
# n_points = 5
# k_space = np.linspace(150, 180, n_points)
# prices = black_formula(F, k_space, vol, T, is_call)
# print(prices)
