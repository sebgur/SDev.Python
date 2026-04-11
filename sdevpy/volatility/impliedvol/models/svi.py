import numpy as np
import numpy.typing as npt
from sdevpy.volatility.impliedvol.impliedvol import ParamSection
from sdevpy.maths import constants


class SviSection(ParamSection):
    def __init__(self, time, check_butterfly=False):
        super().__init__(time, svi_formula)
        self.check_butterfly = check_butterfly
        self.model = 'SVI'

    def check_params(self):
        """ Check parameter consistency """
        return svi_check_params(self.params, self.check_butterfly)

    def dump_params(self):
        """ Dump parameter to dictionary """
        data = {'a': self.params[0], 'b': self.params[1], 'rho': self.params[2],
                'm': self.params[3], 'sigma': self.params[4]}
        return data


def svi(t: float, x: npt.ArrayLike, *params) -> npt.ArrayLike:
    """ SVI formula. Original by J. Gatheral, modified here:
        * x is the log-moneyness, not just the moneyness as in the original
    """
    # Retrieve parameters
    if len(params) != 5:
        raise ValueError(f"Incorrect parameter size in SVI: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Check constraints
    is_ok, _ = svi_check_params(params)
    if not is_ok:
        raise ValueError("Invalid SVI parameters")

    # Calculate
    xm = x - m # x is the log-moneyness
    var = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(var < 0.0):
        raise ValueError("Negative variance in SVI formula")

    vol = np.sqrt(var / t)
    return vol


def svi_check_params(params: list[float], check_butterfly: bool=False) -> tuple[bool, float]:
    """ Check consistency of SVI parameters """
    a = params[0]
    b = params[1]
    rho = params[2]
    # m = params[3] # No particular conditions on m
    sigma = params[4]

    is_ok = True
    # Check constraints
    if np.any(b < 0.0) or np.any(np.abs(rho) >= 1) or np.any(sigma < 0.0):
        is_ok = False

    if is_ok:
        if np.any(a + b * sigma * np.sqrt(1.0 - rho**2) < 0.0):
            is_ok = False

    if is_ok and check_butterfly:
        if b * (1.0 + np.abs(rho)) >= 4.0: # This one we may choose to only enforce during calibration
            is_ok = False

    # Sudden death for now
    penalty = 0.0 if is_ok else constants.FLOAT_INFTY

    return is_ok, penalty


def svi_formula(t: float, x: npt.ArrayLike, params: list[float]) -> npt.ArrayLike:
    """ Wrapper on SVI formula to take parameter vector as input """
    return svi(t, x, *params)


def sample_params(t: float) -> list[float]:
    """ Guess parameters for display or optimization initial point """
    a = 0.25**2 * t
    b = 0.0
    rho, m = 0.0, 0.0
    sigma = 0.0
    return np.array([a, b, rho, m, sigma])


if __name__ == "__main__":
    print("Hello")
