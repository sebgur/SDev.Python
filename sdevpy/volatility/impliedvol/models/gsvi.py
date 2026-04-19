""" Simple reparameterization of the original SVI model of J. Gatheral by incorporating the
    time dependence in the parameters, and using the original SVI formula to express the
    vol squared rather than the variance. This is the parameterization used in the
    Term-Structure SVI models of [Gurrieri2010] in
    See Gurrieri, 'A Class of Term Structures for SVI Implied Volatility', 2010
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1779463
    """
from pathlib import Path
import numpy as np
import numpy.typing as npt
import logging
from sdevpy.maths import constants
log = logging.getLogger(Path(__file__).stem)


def gsvi_formula(x: npt.ArrayLike, params: list[float]) -> npt.ArrayLike:
    """ gSVI formula as in [Gurrieri2010] """
    # Retrieve parameters
    if len(params) != 5:
        raise ValueError(f"Incorrect parameter size in gSVI: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Calculate
    xm = x - m # x is the log-moneyness
    var = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(var < 0.0):
        log.warning("Negative variance in gSVI formula: flooring to 0")
        var = np.maximum(var, 0.0)

    vol = np.sqrt(var)
    return vol


def gsvi_check_params(params: list[float], check_butterfly: bool=False) -> tuple[bool, float]:
    """ Check consistency of gSVI parameters """
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


# if __name__ == "__main__":
#     print("Hello")
