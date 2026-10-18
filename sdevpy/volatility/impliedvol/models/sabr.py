""" Utilities for SABR model, in its original formulation by Hagan in 'Managing Smile Risk',
    Wilmott Magazine """
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.stats import norm


def implied_vol(t: float, k: npt.ArrayLike, f: float, alpha: float, beta: float, nu: float,
                rho: float) -> npt.ArrayLike:
    """ Hagan's original formula (2.17a) in 'Managing Smile risk', Wilmott Magazine. We introduce
        a Taylor expansion around ATM to take care of the singularity. """
    v = np.power(f * k, (1.0 - beta) / 2.0)
    log_m = np.log(f / k)

    # Big numerator piece
    tmp1 = nu * nu * (2.0 - 3.0 * rho * rho) / 24.0
    tmp1 += rho * beta * nu * alpha / (v * 4.0)
    tmp1 += np.power((1.0 - beta) * alpha, 2) / ((v**2) * 24.0)
    tmp1 = alpha * (1.0 + tmp1 * t)

    # Big denominator piece
    big_c = np.power((1.0 - beta) * log_m, 2)
    tmp2 = v * (1.0 + big_c / 24.0 + big_c**2 / 1920.0)

    # Correction
    z = (nu / alpha) * v * log_m
    m_epsilon = 1e-15
    # Unfortunately where() evaluates both so gives warning when going on the wrong side.
    # We temporarily suspend the warning of division by 0.
    z2 = z * z
    with np.errstate(invalid='ignore'):
        correction = np.where(z2 > 10.0 * m_epsilon,
                            z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z2) + z - rho) / (1.0 - rho)),
                            1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z2 / 12.0)

    return tmp1 / tmp2 * correction


def sabr_from_dict(t: float, k: npt.ArrayLike, f: float, parameters: dict) -> npt.ArrayLike:
    """ Hagan's original formula (2.17a) in 'Managing Smile risk', Wilmott Magazine. We introduce
        a Taylor expansion around ATM to take care of the singularity. The parameters are passed
        as the vector [ln_vol, beta, nu, rho] where ln_vol is a more intuitive parameter than
        the original alpha. It has the meaning of a log-normal vol, and we define it through
        alpha = ln_vol * fwd ^ (1.0 - beta) """
    lnvol = parameters['LnVol']
    beta = parameters['Beta']
    nu = parameters['Nu']
    rho = parameters['Rho']
    alpha = calculate_alpha(lnvol, f, beta)
    return implied_vol(t, k, f, alpha, beta, nu, rho)


def calculate_alpha(ln_vol: float, fwd: float, beta: float) -> float:
    """ Calculate original parameter alpha with our definition in terms of ln_vol, i.e.
        alpha = ln_vol * fwd ^ (1.0 - beta) """
    return ln_vol * fwd ** (1.0 - beta)


def alpha_from_atm_nvol(t: float, f: float, nvol: float, beta: float, nu: float, rho: float) -> float:
    """ Closed-form alpha such that the shifted SABR (Hagan lognormal) ATM vol reproduces
        the given ATM normal vol at forward f """
    # Exact ATM Bachelier -> shifted Black conversion
    bs_vol = 2.0 / np.sqrt(t) * norm.ppf(0.5 * (1.0 + nvol * np.sqrt(t / (2.0 * np.pi)) / f))
    # Hagan ATM formula as a cubic in alpha
    c = f ** (1.0 - beta)
    a3 = t * (1.0 - beta)**2 / (24.0 * c**2)
    a2 = t * rho * beta * nu / (4.0 * c)
    a1 = 1.0 + t * (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    roots = np.roots([a3, a2, a1, -bs_vol * c])
    real_roots = roots[np.abs(roots.imag) < 1e-12].real
    return real_roots[real_roots > 0].min()


if __name__ == "__main__":
    # Test near ATM
    expiry = 0.5
    fwd = 0.04
    params = {'LnVol': 0.25, 'Beta': 0.4, 'Nu': 0.50, 'Rho': -0.25}

    min_k = fwd - 0.03
    max_k = fwd + 0.06
    n_points = 100
    strikes = np.linspace(min_k, max_k, n_points)
    vol = sabr_from_dict(expiry, strikes, fwd, params)
    print(vol)

    plt.plot(strikes, vol, color='blue', label='sabr_iv')
    plt.legend(loc='upper right')
    plt.show()
