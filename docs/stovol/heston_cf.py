"""
Heston closed-form (Lewis / Carr-Madan style) for vanilla European calls.
Used as a sanity check for the SLV PDE when the leverage L == 1.
"""
import numpy as np
from scipy.integrate import quad


def heston_call(S0, K, T, r, kappa, theta, xi, rho, v0):
    """Standard Heston call price via the characteristic-function integral."""
    def char_func(phi, j):
        if j == 1:
            u = 0.5; b = kappa - rho * xi
        else:
            u = -0.5; b = kappa
        a = kappa * theta
        d = np.sqrt((rho * xi * 1j * phi - b) ** 2
                    - xi ** 2 * (2 * u * 1j * phi - phi ** 2))
        g = (b - rho * xi * 1j * phi - d) / (b - rho * xi * 1j * phi + d)
        C = (r * 1j * phi * T
             + (a / xi ** 2) * ((b - rho * xi * 1j * phi - d) * T
                                 - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
        D = ((b - rho * xi * 1j * phi - d) / xi ** 2
             * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
        return np.exp(C + D * v0 + 1j * phi * np.log(S0))

    def integrand(phi, j):
        return np.real(np.exp(-1j * phi * np.log(K)) * char_func(phi, j) / (1j * phi))

    P1 = 0.5 + 1.0 / np.pi * quad(integrand, 1e-8, 200, args=(1,), limit=200)[0]
    P2 = 0.5 + 1.0 / np.pi * quad(integrand, 1e-8, 200, args=(2,), limit=200)[0]
    return S0 * P1 - K * np.exp(-r * T) * P2


def heston_put(S0, K, T, r, kappa, theta, xi, rho, v0):
    c = heston_call(S0, K, T, r, kappa, theta, xi, rho, v0)
    return c - S0 + K * np.exp(-r * T)
