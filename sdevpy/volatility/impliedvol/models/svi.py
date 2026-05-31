import numpy as np
import numpy.typing as npt
from sdevpy.maths import constants


def svi(t: float, log_x: npt.ArrayLike, *params) -> npt.ArrayLike:
    """ SVI formula. Original by J. Gatheral, modified here:
        Args:
            - log_x is the log-moneyness, not just the moneyness as in the original
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
        # return np.full_like(np.asarray(x, dtype=float), np.nan)
        raise ValueError("Invalid SVI parameters")

    # Calculate
    log_xm = log_x - m # x is the log-moneyness
    var = a + b * (rho * log_xm + np.sqrt(log_xm**2 + sigma**2))
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


def svi_formula(t: float, log_x: npt.ArrayLike, params: list[float]) -> npt.ArrayLike:
    """ Wrapper on SVI formula to take parameter vector as input.
        log_x is the log-moneyness
    """
    return svi(t, log_x, *params)


def taylor_dlog_x(t: float, log_x: npt.ArrayLike, params: list[float]) -> tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """ Analytical differentiation along the log-moneyness """
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
        # return np.full_like(np.asarray(x, dtype=float), np.nan)
        raise ValueError("Invalid SVI parameters")

    # Calculate vol
    log_xm = log_x - m # x is the log-moneyness
    log_xm2s2 = log_xm**2 + sigma**2
    sqrt_ = np.sqrt(log_xm2s2)
    var = a + b * (rho * log_xm + sqrt_)
    if np.any(var < 0.0):
        raise ValueError("Negative variance in SVI formula")

    vol = np.sqrt(var / t)

    # 1st diff
    dvol_dvar = 0.5 / vol / t
    dvar_dlog_xm = b * (rho + log_xm / sqrt_)
    dvol_dlog_xm = dvol_dvar * dvar_dlog_xm

    # 2nd diff
    d2vol_dvar2 = -0.25 / np.power(vol, 3) / t**2
    dvar2_dlog_xm2 = b * (1.0 - log_xm**2 / log_xm2s2) / sqrt_
    d2vol_dlog_xm2 = d2vol_dvar2 * dvar_dlog_xm**2 + dvol_dvar * dvar2_dlog_xm2

    return vol, dvol_dlog_xm, d2vol_dlog_xm2


def sample_params(t: float) -> list[float]:
    """ Guess parameters for display or optimization initial point """
    a = 0.25**2 * t
    b = 0.0
    rho, m = 0.0, 0.0
    sigma = 0.0
    return np.array([a, b, rho, m, sigma])


if __name__ == "__main__":
    print("Hello")
