from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class ImpliedVol(ABC):
    def value(t, x):
        pass



def SVI(t, x, *params):
    """ SVI formula. Original by J. Gatheral, modified here:
        * x is the log-moneyness, not just the moneyness as in the original
        * the first parameter alnv has the dimension of a lognormal vol
          and the original parameter a is defined as a = alnv**2 * t. This
          is to help optimizations.
    """
    # Retrieve parameters
    if (len(params) != 5):
        raise RuntimeError(f"Incorrect parameter size in SVI: {len(params)}")

    alnv = params[0]
    a = alnv**2 * t
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Check constraints
    if a < 0.0 or b < 0.0 or abs(rho) >= 1 or sigma <= 0.0:
        raise ValueError("SVI basic constraints violated")

    if a + b * sigma * np.sqrt(1.0 - rho**2) < 0.0:
        raise ValueError("SVI positivity constraint violated")

    # if b * (1.0 + abs(rho)) >= 4.0: # This one we may choose to only enforce during calibration
    #     raise ValueError("SVI butterfly constraint violated")

    # Calculate
    xm = x - m
    var = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(var < 0.0):
        raise ValueError("Negative variance in SVI formula")

    vol = np.sqrt(var / t)
    return vol


if __name__ == "__main__":
    alnv = 0.25 # a > 0
    b = 0.2 # b > 0
    rho = 0.0 # -1 < rho < 1
    m = 0.5 # No constraints
    sigma = 0.25 # > 0

    t = 1.5
    k = np.linspace(0.2, 3.0, 100)
    x = np.log(k)
    vol = SVI(t, x, alnv, b, rho, m, sigma)

    plt.plot(k, vol)
    plt.show()
