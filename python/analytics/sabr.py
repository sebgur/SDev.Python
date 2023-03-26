# import sys
import numpy as np
# import matplotlib.pyplot as plt
# import tools.utils as utils
# from keras.models import load_model
# from sklearn.externals import joblib
# import pandas as pd
# from scipy.stats import norm
# from scipy.interpolate import CubicSpline


# Intermediate function for Hagan's SABR
def chi(z, rho):
    eps_sabr = 0.0001

    tmp1 = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    np.where(abs(z) < eps_sabr, z, z)

    zz = np.where((tmp1 + z - rho > 0.0),
                  np.log((tmp1 + z - rho) / (1.0 - rho)), np.log((1.0 + rho) / (tmp1 - (z - rho))))

    return zz


# Hagan's SABR formula
def sabr_iv2(alpha, beta, nu, rho, f, k, t):
    eps_sabr = 0.0001
    v = (f * k) ** ((1.0 - beta) / 2.0)
    log_fk = np.log(f / k)
    tmp1 = nu * nu * (2.0 - 3.0 * rho * rho) / 24.0
    tmp1 += rho * beta * nu * alpha / (v * 4.0)
    tmp1 += (1.0 - beta) * (1.0 - beta) * alpha * alpha / ((v ** 2) * 24.0)
    tmp1 = alpha * (1.0 + (tmp1 * t))
    tmp2 = v * (1.0 + np.power(log_fk * (1.0 - beta), 2.0) / 24.0 + np.power(log_fk * (1.0 - beta), 4.0)
                / 1920.0)
    z = nu / alpha * v * log_fk
    chi_z = chi(z, rho)
    vol = np.where(abs(f - k) > eps_sabr, tmp1 / tmp2 * z / chi_z, tmp1 / np.power(f, 1.0 - beta))
    return vol


# Hagan's SABR formula with ATM limit handling
def sabr_iv(alpha, beta, nu, rho, f, k, t):
    big_a = np.power(f * k, 1.0 - beta)
    sqrt_a = np.sqrt(big_a)
    log_m = np.log(f / k)

    z = (nu / alpha) * sqrt_a * log_m
    m_epsilon = 1e-15
    # if z * z > 10.0 * m_epsilon:
    #     multiplier = z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    # else:
    #     multiplier = 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0

    # The below is for vectors but where() evaluates both so gives warning when going on the wrong side
    multiplier = np.where(z * z > 10.0 * m_epsilon,
                          z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho)),
                          1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0)

    big_c = np.power((1 - beta) * log_m, 2)
    big_d = sqrt_a * (1.0 + big_c / 24.0 + np.power(big_c, 2) / 1920.0)
    d = 1.0 + t * (np.power((1 - beta) * alpha, 2) / (24.0 * big_a)
                   + 0.25 * rho * beta * nu * alpha / sqrt_a
                   + (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0))

    return (alpha / big_d) * multiplier * d
