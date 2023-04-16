""" Utilities for SABR model, in its original formulation by Hagan in 'Managing Smile Risk',
    Wilmott Magazine """
import numpy as np
import matplotlib.pyplot as plt


def implied_vol(t, k, f, alpha, beta, nu, rho):
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
    np.seterr(invalid='ignore')
    z2 = z * z
    correction = np.where(z2 > 10.0 * m_epsilon,
                          z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z2) + z - rho) / (1.0 - rho)),
                          1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z2 / 12.0)
    np.seterr(invalid='warn')

    return tmp1 / tmp2 * correction


def implied_vol_vec(t, k, f, parameters):
    """ Hagan's original formula (2.17a) in 'Managing Smile risk', Wilmott Magazine. We introduce
        a Taylor expansion around ATM to take care of the singularity. The parameters are passed
        as the vector [ln_vol, beta, nu, rho] where ln_vol is a more intuitive parameter than
        the original alpha. It has the meaning of a log-normal vol, and we define it through
        alpha = ln_vol * fwd ^ (1.0 - beta) """
    # lnvol = parameters.lnvol
    # beta = parameters.beta
    # nu = parameters.nu
    # rho = parameters.rho
    lnvol = parameters['LnVol']
    beta = parameters['Beta']
    nu = parameters['Nu']
    rho = parameters['Rho']
    alpha = calculate_alpha(lnvol, f, beta)
    return implied_vol(t, k, f, alpha, beta, nu, rho)


def calculate_alpha(ln_vol, fwd, beta):
    """ Calculate original parameter alpha with our definition in terms of ln_vol, i.e.
        alpha = ln_vol * fwd ^ (1.0 - beta) """
    return ln_vol * fwd ** (1.0 - beta)


if __name__ == "__main__":
    # Test near ATM
    EXPIRY = 0.5
    FWD = 0.04
    PARAMS = [0.25, 0.4, 0.50, -0.25]

    MIN_STRIKE = FWD - 0.03
    MAX_STRIKE = FWD + 0.06
    NUM_POINTS = 100
    strikes = np.linspace(MIN_STRIKE, MAX_STRIKE, NUM_POINTS)
    vol = implied_vol_vec(EXPIRY, strikes, FWD, PARAMS)

    plt.plot(strikes, vol, color='blue', label='sabr_iv')
    plt.legend(loc='upper right')
    plt.show()
