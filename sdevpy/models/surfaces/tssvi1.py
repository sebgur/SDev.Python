""" Term-structure model for SVI. Give each SVI parameter a parametric formula along time and
    enforce no-arbitrage (approximately). This model has 11 parameters.
    See Gurrieri, 'A Class of Term Structures for SVI Implied Volatility', 2010
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1779463
"""
import numpy as np
import numpy.typing as npt
import datetime as dt
from sdevpy.models.surfaces.zerosurface import TermStructureParametricZeroSurface
from sdevpy.models.svi import svi_formula
from sdevpy.market import eqvolsurface as vsurf
from sdevpy.models.surfaces.optionsurface import calibration_targets
from sdevpy.tools import timegrids


################## TODO ###########################################
# Do the calibration


class TsSvi1(TermStructureParametricZeroSurface):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_params = 11
        self.calculable_at_zero = False
        self.tmax = kwargs.get('tmax', 42)

    def formula(self, t: float, k: npt.ArrayLike, is_call: bool, f: npt.ArrayLike,
                params: list[float]) -> npt.ArrayLike:
        # Calculate log-moneyness
        log_m = np.log(k / f)
        # print(f"time: {t}")
        # print(f"x: {log_m}")
        # print(f"params: {params}")
        vol = svi_formula(t, log_m, params)
        return vol

    def formula_parameters(self, t: npt.ArrayLike, params: list[float]) -> list[float]:
        """ Calculate parameters according to the TsSvi1 formulas """
        if np.any(t < self.eps):
            raise ValueError("TsSvi1 is not calculable at t = 0")

        # Get parameters
        v0, vinf, b_, tau, alpha, beta, r, x0star, lambda0, gamma, delta = self.get_parameters(params)
        if r < -1.0 or r > 1.0:
            raise ValueError("Correlation should be between -1 and 1 in TsSvi1")

        if delta + 1.0 < self.eps:
            raise ValueError("Delta should be stricktly higher than -1 in TsSvi1")

        # Calculate new variables
        one_minus_rho2 = 1.0 - r * r
        if one_minus_rho2 < 0.0:
            raise ValueError("Correlation should be between -1 and 1 in TsSvi1")

        # Vectorize r (the other ones broadcast fine thanks to t)
        t = np.asarray(t, dtype=float)
        r = np.full_like(t, r)

        pow_ = beta + delta
        vstar = alpha * gamma * one_minus_rho2 / (pow_ + 1.0) * np.power(t, pow_)
        f0 = v0 * v0
        finf = vinf * vinf
        vstar += -b_ * tau + finf + tau / t * (b_ * (t + tau) + f0 - finf) * (1.0 - np.exp(-t / tau))
        b = alpha * np.power(t, beta - 1.0)

        lambda_ = lambda0 + gamma / (delta + 1.0) * np.power(t, delta + 1.0)
        xstar = x0star - r * (lambda_ - lambda0)

        # Go back to standard parameters
        a = vstar - b * lambda_ * one_minus_rho2
        m = xstar + r * lambda_
        s = lambda_ * np.sqrt(one_minus_rho2)

        return [a, b, r, m, s]

    def get_parameters(self, x: list[float]) -> tuple[float, ...]:
        """ Return named parameters from input list """
        if len(x) != 11:
            raise ValueError("The number of parameters should be 11 for TsSvi1")

        return x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]


if __name__ == "__main__":
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)

    # Retrieve target market option data
    file = vsurf.data_file(vsurf.test_data_folder(), name, valdate)
    surface_data = vsurf.eqvolsurfacedata_from_file(file)
    expiries = surface_data.expiries
    fwds = surface_data.forwards
    strike_surface = surface_data.get_strikes('absolute')
    vol_surface = surface_data.vols

    # Set calibration time grid
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

    # Set calibration targets
    cf_price_surface, ftols = calibration_targets(expiry_grid, fwds, strike_surface, vol_surface)

    # Set up the model
    surface = TsSvi1()
    params = [0.20, 0.25, 0.10, 2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    surface.calibrated_parameters = params

    t = expiry_grid
    k = strike_surface
    f = fwds
    # t = np.asarray([0.5, 1.5, 2.5])
    # k = np.asarray([90, 100, 110])
    # f = np.asarray([95, 105, 115])
    is_call = True
    z = surface.calculate(t, k, is_call, f)
    print(f"Result shape: {z.shape}")
    print(f"Result {z}")

    # Estimate model on points and calculate RMSE, plot comparison
