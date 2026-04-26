""" VSVI stands for 'Vol SVI'. It is purely a parametric model, without strong financial
    inspiration, as opposed to its original SVI. This model simply gives an SVI-like
    parametric shape to the volatility, as opposed to the variance in the original paper
    by J. Gatheral. """
import datetime as dt
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from scipy.stats import norm
from sdevpy.volatility.localvol.localvol import ParamLocalVolSection
from sdevpy.maths import constants


DFLT_PERCENTS = [0.10, 0.25, 0.50, 0.75, 0.90]


def create_section(time: float, param_config: dict=None, fill_sample: bool=True) -> ParamLocalVolSection:
    """ Create a vSVI section """
    section = VSviSection(time)
    if param_config is None and fill_sample:
        params = sample_params(time) # Fill with sample
    else:
        params = []
        params.append(param_config['a'])
        params.append(param_config['b'])
        params.append(param_config['rho'])
        params.append(param_config['m'])
        params.append(param_config['sigma'])

    section.update_params(params)
    return section


class VSviSection(ParamLocalVolSection):
    def __init__(self, time):
        super().__init__(time, vsvi_formula)
        self.model = 'vSVI'

    def check_params(self):
        return vsvi_check_params(self.params)

    def dump_params(self):
        data = {'a': self.params[0], 'b': self.params[1], 'rho': self.params[2],
                'm': self.params[3], 'sigma': self.params[4]}
        return data

    def constraints(self):
        # a, b, rho, m, sigma
        lw_bounds = [0.01, -0.1, -0.99, -0.1, 0.01]
        up_bounds = [1.0, 0.1, 0.99, 0.1, 1.0]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)
        return bounds

def vsvi(x, *params):
    """ SVI-like formula but applied to the vol directly. This no longer has
        the original features of no-arbitrage and the interpretation as a limit
        of Heston. It is purely used for its parametric shape here.
    """
    # Retrieve parameters
    if len(params) != 5:
        raise ValueError(f"Incorrect parameter size in vSVI: {len(params)}")

    a = params[0]
    b = params[1]
    rho = params[2]
    m = params[3]
    sigma = params[4]

    # Check constraints
    is_ok, _ = vsvi_check_params(params)
    if not is_ok:
        raise ValueError("Invalid vSVI parameters")

    # Calculate
    xm = x - m # x is the log-moneyness
    vol = a + b * (rho * xm + np.sqrt(xm**2 + sigma**2))
    if np.any(vol < 0.0):
        raise ValueError("Negative variance in vSVI formula")

    return vol


def vsvi_check_params(params: list[float]):
    """ Check parameters of the vSVI model """
    a = params[0]
    b = params[1]
    rho = params[2]
    # m = params[3] # No constraints on m
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


def vsvi_formula(t: float, x: npt.ArrayLike, params: list[float]) -> npt.ArrayLike:
    """ Wrapper on vSVI formula to take parameter vector as input """
    return vsvi(x, *params)


def sample_params(t: float, vol: float=0.25) -> npt.ArrayLike:
    """ Guess parameters for display or optimization initial point """
    a = vol
    b = 0.1 / t
    rho = -0.25
    m = 0.0
    sigma = 0.25 * t
    return np.array([a, b, rho, m, sigma])


def generate_sample_data(valdate: dt.datetime, terms, base_vol: float=0.25,
                         percents: list[float]=DFLT_PERCENTS):
    """ Generate sample data for the vSVI model """
    spot, r, q = 100.0, 0.04, 0.02

    expiries, fwds, strike_surface, vol_surface = [], [], [], []
    for term in terms:
        expiry = valdate + dt.timedelta(days=int(term * 365.25))
        fwd = spot * np.exp((r - q) * term)
        base_std = base_vol * np.sqrt(term)
        a, b, rho, m, sigma = base_vol, 0.1 / term, -0.25, 0.0, 0.25 * term # a, b, rho, m, sigma
        strikes, vols = [], []
        for p in percents:
            logm = -0.5 * base_std**2 + base_std * norm.ppf(p)
            strikes.append(fwd * np.exp(logm))
            # a, b, rho, m, sigma = 0.179, 5.3, -0.40, -0.019, 0.025
            vols.append(vsvi(logm, a, b, rho, m, sigma))

        expiries.append(expiry)
        fwds.append(fwd)
        strike_surface.append(strikes)
        vol_surface.append(vols)

    return np.array(expiries), np.array(fwds), np.array(strike_surface), np.array(vol_surface)


if __name__ == "__main__":
    print("Hello")
