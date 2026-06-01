""" Payoff smoothers for numerical methods """
import numpy as np
import numpy.typing as npt


# Smoothing parameters (for the max function)
SMOOTH_VOL = 0.40
SMOOTH_TIME = 10.0 / 365.0
SMOOTH_STDEV = SMOOTH_VOL * np.sqrt(SMOOTH_TIME)


def approx_cdf(x: npt.ArrayLike) -> npt.ArrayLike:
    """ Simple approximation of CDF """
    return 1.0 / (1.0 + np.exp(-x / 0.5879))


def smooth_max_diff(spot: npt.ArrayLike, strike: npt.ArrayLike) -> npt.ArrayLike:
    """ Smoothing of call payoff """
    d1 = np.log(spot / strike) / SMOOTH_STDEV + 0.5 * SMOOTH_STDEV
    d2 = d1 - SMOOTH_STDEV
    n1 = approx_cdf(d1)
    n2 = approx_cdf(d2)
    return spot * n1 - 0.5 * strike * (n1 + n2)  # Average
