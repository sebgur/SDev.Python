import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def bachelier_price(expiry, fwd, strike, vol):
    stdev = vol * expiry**0.5
    d = (fwd - strike) / stdev
    price = stdev * (d * norm.cdf(d) + norm.pdf(d))
    return price


# Direct method by numerical inversion using Brent
def bachelier_iv_solve(expiry, fwd, strike, price):
    options = {'xtol': 1e-4, 'maxiter': 100, 'disp': False}
    xmin = 1e-6
    xmax = 1.0

    def error(vol):
        premium = bachelier_price(expiry, fwd, strike, vol)
        return (premium - price) ** 2

    res = minimize_scalar(fun=error, bracket=(xmin, xmax), options=options, method='brent')

    return res.x


# P. Jaeckel's method in "Implied Normal Volatility", 6th Jun. 2017
def bachelier_iv(expiry, fwd, strike, price, is_call=True):
    # Special case at ATM
    if np.abs(fwd - strike) < 1e-8:
        return price * np.sqrt(2.0 * np.pi) / np.sqrt(expiry)

    # General case
    tilde_phi_star_c = -0.001882039271
    theta = 1.0 if is_call else -1.0

    tilde_phi_star = -np.abs(price - np.maximum(theta * (fwd - strike), 0.0)) / np.abs(fwd - strike)
    em5 = 1e-5

    if tilde_phi_star < tilde_phi_star_c:
        g = 1.0 / (tilde_phi_star - 0.5)
        g2 = g**2
        em3 = 1e-3
        num = 0.032114372355 - g2 * (0.016969777977 - g2 * (2.6207332461 * em3 - 9.6066952861 * em5 * g2))
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
    sigma = np.abs(fwd - strike) / (np.abs(xs) * np.sqrt(expiry))
    return sigma
