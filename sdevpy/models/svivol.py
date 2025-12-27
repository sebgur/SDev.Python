import numpy as np
import matplotlib.pyplot as plt
from sdevpy.models.impliedvol import ParamSection
from sdevpy.maths import constants


class SviVolSection(ParamSection):
    def __init__(self):
        super().__init__(svivol_formula)

    def check_params(self):
        return svivol_check_params(self.params)


def svivol(t, x, *params):
    """ SVI-like formula but applied to the vol directly. This no longer has
        the original features of no-arbitrage and the interpretation as a limit
        of Heston. It is purely used for its parametric shape here.
    """
    # Retrieve parameters
    if (len(params) != 5):
        raise RuntimeError(f"Incorrect parameter size in SviVol: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Check constraints
    is_ok, _ = svivol_check_params(params)
    if not is_ok:
        raise RuntimeError("Invalid SviVol parameters")

    # Calculate
    xm = x - m # x is the log-moneyness
    vol = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(vol < 0.0):
        raise ValueError("Negative variance in SVI formula")

    return vol


def svivol_check_params(params):
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

    # Sudden death for now
    penalty = 0.0 if is_ok else constants.FLOAT_INFTY

    return is_ok, penalty


def svivol_formula(t, x, params):
    """ Wrapper on SVI formula to take parameter vector as input """
    return svivol(t, x, *params)


def sample_params(t):
    """ Guess parameters for display or optimization initial point """
    a = 0.25
    b = 0.0
    rho, m = 0.0, 0.0
    sigma = 0.0
    return np.array([a, b, rho, m, sigma])


if __name__ == "__main__":
    t = 1/365
    a = 0.25 # a > 0
    b = 0.01 #a / np.log(2) # b > 0
    rho = 0.0 # -1 < rho < 1
    m = 0.0 # No constraints
    sigma = 0.5 #0.5 / np.sqrt(t) # > 0
    params = [a, b, rho, m, sigma]

    k = np.linspace(0.2, 3.0, 100)
    x = np.log(k)

    vol = svivol_formula(t, x, params)
    print(svivol_formula(t, m, params))
    plt.plot(x, vol)
    plt.show()
