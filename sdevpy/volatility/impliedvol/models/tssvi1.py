""" Term-structure model for SVI. Give each SVI parameter a parametric formula along time and
    enforce no-arbitrage (approximately). This model has 11 parameters.
    See Gurrieri, 'A Class of Term Structures for SVI Implied Volatility', 2010
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1779463
"""
import numpy as np
import numpy.typing as npt
import datetime as dt
import scipy.optimize as opt
from sdevpy.volatility.impliedvol.zerosurface import TermStructureParametricZeroSurface
from sdevpy.volatility.impliedvol.models import gsvi
from sdevpy.market import eqvolsurface as vsurf
from sdevpy.tools import timegrids
from sdevpy.maths.metrics import rmse
from sdevpy.volatility.impliedvol.impliedvol_calib import TsIvCalibrator


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
        vol = gsvi.gsvi_formula(log_m, params)
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
            raise ValueError("Delta should be strictly higher than -1 in TsSvi1")

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

    def check_params(self):
        """ Check validity of the parameters """
        sample_times = np.asarray([1 / 365, 7 / 365, 30 / 365, 0.5, 1, 5, 10, 40])
        sample_params = self.formula_parameters(sample_times, self.params)
        is_ok, penalty = gsvi.gsvi_check_params(sample_params)
        return is_ok, penalty

    def bounds(self, keep_feasible: bool=False):
        """ Recommended bounds """
        lw_bounds = [0.0, 0.00001, -1.0,  0.1, 0.0000, 0.0001, -0.99, -0.50, 0.0, 0.0, -0.999]
        up_bounds = [1.0, 1.00000,  1.0, 50.0, 5.0000, 0.9990,  0.20,  2.00, 2.0, 5.0,  5.000]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=keep_feasible)
        return bounds

    def initial_point(self):
        """ Recommended initial point """
        init_point = [0.1, 0.10000, -0.1,  1.0, 0.1000, 0.1000, -0.30,  0.10, 0.1, 1.0,  1.000]
        return np.asarray(init_point)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)

    # Retrieve target market option data
    file = vsurf.data_file(name, valdate)
    mkt_data = vsurf.eqvolsurfacedata_from_file(file)
    expiries = mkt_data.expiries
    fwds = mkt_data.forwards
    strike_surface = mkt_data.get_strikes('absolute')
    vol_surface = mkt_data.vols

    # Initialize model
    model = TsSvi1()

    # Calibrate model
    calibrator = TsIvCalibrator(model, {'optimizer': 'SLSQP', 'tol': 1e-6})
    calibrator.calibrate(mkt_data)

    # Estimate model on points and calculate RMSE, plot comparison
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            exp_idx = n_cols * i + j
            # expiry = expiry_grid[exp_idx]
            expiry = timegrids.model_time(valdate, expiries[exp_idx])
            fwd = fwds[exp_idx]
            strikes = strike_surface[exp_idx]
            min_k, max_k = strikes[0], strikes[-1]
            m_strikes = np.linspace(0.8 * min_k, 1.2 * max_k, 100)
            m_vols = model.calculate(expiry, m_strikes, True, fwd)
            ax.scatter(strikes, vol_surface[exp_idx], label="market", color='black')
            ax.plot(m_strikes, m_vols, label="model", color='green')
            model_vols = model.calculate(expiry, strikes, True, fwd)
            vol_rmse = rmse(vol_surface[exp_idx], model_vols)
            ax.set_title(f"T:{expiry:.2f}, RMSE(bps): {10000.0 * vol_rmse:,.2f}")
            ax.set_xlabel('strike')
            ax.set_ylabel('vol')
            ax.legend()

    fig.suptitle('Option vols, Model vs Market', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
