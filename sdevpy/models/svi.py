import numpy as np
import matplotlib.pyplot as plt
from sdevpy.models.impliedvol import ParamSection
from sdevpy.maths import constants


class SviSection(ParamSection):
    def __init__(self, check_butterfly=False):
        super().__init__(svi_formula)
        self.check_butterfly = check_butterfly

    def check_params(self):
        return svi_check_params(self.params, self.check_butterfly)


def svi(t, x, *params):
    """ SVI formula. Original by J. Gatheral, modified here:
        * x is the log-moneyness, not just the moneyness as in the original
    """
    # Retrieve parameters
    if (len(params) != 5):
        raise RuntimeError(f"Incorrect parameter size in SVI: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Check constraints
    is_ok, _ = svi_check_params(params)
    if not is_ok:
        raise RuntimeError("Invalid SVI parameters")

    # Calculate
    xm = x - m # x is the log-moneyness
    var = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(var < 0.0):
        raise ValueError("Negative variance in SVI formula")

    vol = np.sqrt(var / t)
    return vol


def svi_check_params(params, check_butterfly=False):
    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    is_ok = True
    # Check constraints
    if a < 0.0 or b < 0.0 or abs(rho) >= 1 or sigma < 0.0:
        is_ok = False

    if is_ok:
        if a + b * sigma * np.sqrt(1.0 - rho**2) < 0.0:
            is_ok = False

    if is_ok and check_butterfly:
        if b * (1.0 + abs(rho)) >= 4.0: # This one we may choose to only enforce during calibration
            is_ok = False

    # Sudden death for now
    penalty = 0.0 if is_ok else constants.FLOAT_INFTY

    return is_ok, penalty


def svi_formula(t, x, params):
    """ Wrapper on SVI formula to take parameter vector as input """
    return svi(t, x, *params)


def sample_params(t):
    """ Guess parameters for display or optimization initial point """
    a = 0.25**2 * t
    b = 0.0
    rho, m = 0.0, 0.0
    sigma = 0.0
    return np.array([a, b, rho, m, sigma])


if __name__ == "__main__":
    t = 1/365
    alnv = 0.25
    a = alnv**2 * t # a > 0
    b = 0.0 #a / np.log(2) # b > 0
    rho = 0.0 # -1 < rho < 1
    m = 0.0# No constraints
    sigma = 0.0 #0.5 / np.sqrt(t) # > 0
    params = [a, b, rho, m, sigma]

    k = np.linspace(0.2, 3.0, 100)
    x = np.log(k)

    vol = svi_formula(t, x, params)
    print(svi_formula(t, m, params))
    plt.plot(x, vol)
    plt.show()
