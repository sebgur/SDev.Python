""" Utilities for Black-Scholes model """
import scipy.stats
import numpy as np

N = scipy.stats.norm.cdf
# Ninv = scipy.stats.norm.ppf


def price(expiry, strike, is_call, fwd, vol):
    """ Option price under the Black-Scholes model """
    w = 1.0 if is_call else -1.0
    s = vol * np.sqrt(expiry)
    d1 = np.log(fwd / strike) / s + 0.5 * s
    d2 = d1 - s
    return w * (fwd * N(w * d1) - strike * N(w * d2))


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
    NUM_POINTS = 5
    f_space = np.linspace(100, 120, NUM_POINTS)
    k_space = np.linspace(150, 180, NUM_POINTS)
    prices = price(f_space, k_space, VOL, EXPIRY, IS_CALL)
    print(prices)
