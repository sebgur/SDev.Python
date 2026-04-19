""" Utilities for Bachelier model """
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def price(expiry: npt.ArrayLike, strike: npt.ArrayLike, is_call: npt.ArrayLike, fwd: npt.ArrayLike,
          vol: npt.ArrayLike) -> npt.ArrayLike:
    """ Option price under the Bachelier model """
    stdev = vol * expiry**0.5
    d = (fwd - strike) / stdev
    wd = np.where(is_call, d, -d)
    # wd = d if is_call else -d
    return stdev * (wd * norm.cdf(wd) + norm.pdf(d))


def price_straddles(expiry: npt.ArrayLike, strike: npt.ArrayLike, fwd: npt.ArrayLike,
                    vol: npt.ArrayLike) -> npt.ArrayLike:
    """ Straddle price under the Bachelier model.
        Note: we could improve the speed by writing the code in-line instead of
              calling the price() function twice """
    expiries_ = np.asarray(expiry).reshape(-1, 1)
    prices = []
    for i, exp_row in enumerate(expiries_):
        k_prices = []
        for j, k in enumerate(strike[i]):
            iv = vol[i, j]
            call_price = price(exp_row, k, True, fwd, iv)
            put_price = price(exp_row, k, False, fwd, iv)
            k_prices.append(call_price[0] + put_price[0])
        prices.append(k_prices)

    return np.asarray(prices)


def implied_vol_jaeckel(expiry: float, strike: float, is_call: bool, fwd: float, fwd_price: float) -> float:
    """ P. Jaeckel's method in "Implied Normal Volatility", 6th Jun. 2017.
        This is the original, non-vectorized. """
    m = fwd - strike
    abs_m = np.abs(m)
    # Special case at ATM
    if abs_m < 1e-8:
        return fwd_price * np.sqrt(2.0 * np.pi) / np.sqrt(expiry)

    # General case
    tilde_phi_star_c = -0.001882039271
    theta = 1.0 if is_call else -1.0

    tilde_phi_star = -np.abs(fwd_price - np.maximum(theta * m, 0.0)) / abs_m
    em5 = 1e-5

    if tilde_phi_star < tilde_phi_star_c:
        g = 1.0 / (tilde_phi_star - 0.5)
        g2 = g**2
        em3 = 1e-3
        num = 0.032114372355 - g2 * (0.016969777977 - g2 * (2.6207332461 * em3
                                                             - 9.6066952861 * em5 * g2))
        den = 1.0 - g2 * (0.6635646938 - g2 * (0.14528712196 - 0.010472855461 * g2))
        eta_bar = num / den
        xb = g * (eta_bar * g2 + 1.0 / np.sqrt(2.0 * np.pi))
    else:
        h = np.sqrt(-np.log(-tilde_phi_star))
        num = 9.4883409779 - h * (9.6320903635 - h * (0.58556997323 + 2.1464093351 * h))
        den = 1.0 - h * (0.65174820867 + h * (1.5120247828 + 6.6437847132 * em5 * h))
        xb = num / den

    q = (norm.cdf(xb) + norm.pdf(xb) / xb - tilde_phi_star) / norm.pdf(xb)
    xb2 = xb**2
    num = 3.0 * q * xb2 * (2.0 - q * xb * (2.0 + xb2))
    den = 6.0 + q * xb * (-12.0 + xb * (6.0 * q + xb * (-6.0 + q * xb * (3.0 + xb2))))
    xs = xb + num / den
    sigma = abs_m / (np.abs(xs) * np.sqrt(expiry))
    return sigma


def implied_vol(expiry: npt.ArrayLike, strike: npt.ArrayLike, is_call: npt.ArrayLike,
                fwd: npt.ArrayLike, fwd_price: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """ P. Jaeckel's method in "Implied Normal Volatility", 6th Jun. 2017.
        Vectorized version (vectorization not in Jaeckel). """
    strike = np.asarray(strike, dtype=float)
    fwd_price = np.asarray(fwd_price, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    #### ATM branch ####
    m = fwd - strike
    abs_m = np.abs(m)
    atm_mask = (abs_m < 1e-8)
    atm_result = fwd_price * np.sqrt(2.0 * np.pi / expiry)

    #### General case ####
    tilde_phi_star_c = -0.001882039271
    theta = np.where(is_call, 1.0, -1.0)
    tilde_phi_star = -np.abs(fwd_price - np.maximum(theta * m, 0.0)) / np.where(atm_mask, 1.0, abs_m)
    em5 = 1e-5
    mask_low = (tilde_phi_star < tilde_phi_star_c)

    # Branch = lower
    g = 1.0 / (tilde_phi_star - 0.5)
    g2 = g**2
    em3 = 1e-3
    num_low = 0.032114372355 - g2 * (0.016969777977 - g2 * (2.6207332461 * em3 - 9.6066952861 * em5 * g2))
    den_low = 1.0 - g2 * (0.6635646938 - g2 * (0.14528712196 - 0.010472855461 * g2))
    eta_bar = num_low / den_low
    xb_low = g * (eta_bar * g2 + 1.0 / np.sqrt(2.0 * np.pi))

    # Branch = higher
    # Note: substitute a safe value where mask_low is True to avoid sqrt(log(large))
    safe_tps = np.where(mask_low, -0.001, tilde_phi_star)
    h = np.sqrt(-np.log(-safe_tps))
    num_high = 9.4883409779 - h * (9.6320903635 - h * (0.58556997323 + 2.1464093351 * h))
    den_high = 1.0 - h * (0.65174820867 + h * (1.5120247828 + 6.6437847132 * em5 * h))
    xb_high = num_high / den_high

    # Join lower/higher
    xb = np.where(mask_low, xb_low, xb_high)
    q = (norm.cdf(xb) + norm.pdf(xb) / xb - tilde_phi_star) / norm.pdf(xb)
    xb2 = xb**2
    num = 3.0 * q * xb2 * (2.0 - q * xb * (2.0 + xb2))
    den = 6.0 + q * xb * (-12.0 + xb * (6.0 * q + xb * (-6.0 + q * xb * (3.0 + xb2))))
    xs = xb + num / den
    sigma = abs_m / (np.abs(xs) * np.sqrt(expiry))

    # Join ATM/OTM
    return np.where(atm_mask, atm_result, sigma)


def implied_vol_solve(expiry: float, strike: float, is_call: bool, fwd: float, fwd_price: float) -> float:
    """ Direct method by numerical inversion using Brent """
    options = {'xtol': 1e-4, 'maxiter': 100, 'disp': False}
    xmin = 1e-6
    xmax = 1.0

    def error(vol):
        premium = price(expiry, strike, is_call, fwd, vol)
        return (premium - fwd_price) ** 2

    res = minimize_scalar(fun=error, bracket=(xmin, xmax), options=options, method='brent')

    return res.x
