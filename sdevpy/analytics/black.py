""" Utilities for Black-Scholes model """
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sdevpy.utilities.tools import isiterable


def price(expiry: npt.ArrayLike, strike: npt.ArrayLike, is_call: npt.ArrayLike, fwd: npt.ArrayLike,
          vol: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """ Option price under the Black-Scholes model """
    # w = 1.0 if is_call else -1.0
    w = np.where(is_call, 1.0, -1.0)
    s = vol * np.sqrt(expiry)
    d1 = np.log(fwd / strike) / s + 0.5 * s
    d2 = d1 - s
    return w * (fwd * norm.cdf(w * d1) - strike * norm.cdf(w * d2))


def implied_vol(expiry: float, strike: float, is_call: bool, fwd: float, fwd_price: float) -> float:
    """ Direct method by numerical inversion using Brent.
        Non-vectorized due to solver. """
    options = {'xtol': 1e-4, 'maxiter': 100, 'disp': False}
    xmin = 1e-6
    xmax = 1.0

    def error(vol):
        premium = price(expiry, strike, is_call, fwd, vol)
        return (premium - fwd_price) ** 2

    res = minimize_scalar(fun=error, bracket=(xmin, xmax), options=options, method='brent')
    return res.x


def implied_vols(expiry: float, strike: npt.ArrayLike, is_call: bool, fwd: float,
                 fwd_price: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """ Black implied volatility for vector of strikes/prices """
    if isiterable(strike) and isiterable(fwd_price):
        ivs = [implied_vol(expiry, k, is_call, fwd, p) for k, p in zip(strike, fwd_price, strict=True)]
        return np.asarray(ivs)
    elif not isiterable(strike) and not isiterable(fwd_price):
        return implied_vol(expiry, strike, is_call, fwd, fwd_price)
    else:
        raise ValueError("Incompatible shapes between strikes and prices")


def implied_vol_newton(expiry: float, strike: npt.ArrayLike, is_call: bool, fwd: float,
                       fwd_price: npt.ArrayLike, tol: float=1e-8, max_iter: int=50) -> npt.NDArray[np.float64]:
    """ Using vectorized Newton-Raphson, with faster convergence than Brent.
        However, this method can struggle for very small vegas, so we may want to switch
        to another method (maybe Brent above) below a certain vega threshold.
        Or to switch to Halley's method.
        To be investigated if speed becomes a bottleneck or Brent's method has issues. """
    strike = np.asarray(strike, dtype=float)
    fwd_price = np.asarray(fwd_price, dtype=float)
    vol = np.full_like(fwd_price, 0.25) # Initial guess: flat 25%
    sqrt_t = np.sqrt(expiry)
    for _ in range(max_iter):
        s = vol * sqrt_t
        d1 = np.log(fwd / strike) / s + 0.5 * s
        vega = fwd * norm.pdf(d1) * sqrt_t
        diff = price(expiry, strike, is_call, fwd, vol) - fwd_price
        vol -= diff / vega
        vol = np.maximum(vol, 1e-8) # Keep vol positive
        if np.all(np.abs(diff) < tol):
            break

    # # if len(strike) == 1 and len(fwd_price) == 1 and len(vol) == 1: # Return a scalar if inputs are scalar
    # #     return vol[0]
    # # else:
    # #     return vol
    # return (vol.item() if vol.ndim == 0 or vol.size ==1 else vol)
    return vol


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
