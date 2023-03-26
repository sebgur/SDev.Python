import numpy as np


# Hagan's SABR formula
def sabr_iv(t, k, f, alpha, beta, nu, rho):
    """ Hagan's original formula (2.17a) in 'Managing Smile risk', Wilmott Magazine.
        We introduce a Taylor expansion around ATM to take care of the singularity. """
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
    # Unfortunately where() evaluates both so gives warning when going on the wrong side. We temporarily
    # suspect the warning of division by 0.
    np.seterr(invalid='ignore')
    correction = np.where(z * z > 10.0 * m_epsilon,
                          z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho)),
                          1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0)
    np.seterr(invalid='warn')

    return tmp1 / tmp2 * correction


# Test near ATM
# import matplotlib.pyplot as plt
# beta_ = 0.4
# nu_ = 0.50
# rho_ = -0.25
# fwd = 0.04
# expiry = 0.5
# sigma = 0.25
# alpha_ = sigma * fwd**(1.0 - beta_)
#
# min_strike = fwd - 0.03
# max_strike = fwd + 0.06
# num_points = 100
# strikes = np.linspace(min_strike, max_strike, num_points)
# vol = sabr_iv(expiry, strikes, fwd, alpha_, beta_, nu_, rho_)
#
# plt.plot(strikes, vol, color='blue', label='sabr_iv')
# plt.legend(loc='upper right')
# plt.show()
