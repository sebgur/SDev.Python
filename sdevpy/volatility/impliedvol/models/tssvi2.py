""" Term-structure model for SVI. Give each SVI parameter a parametric formula along time and
    enforce no-arbitrage (approximately). This model has 15 parameters.
"""
import numpy as np
import numpy.typing as npt
import datetime as dt
import scipy.optimize as opt
from sdevpy.volatility.impliedvol.impliedvol import ParametricImpliedVol
from sdevpy.volatility.impliedvol.models import svi
from sdevpy.market import eqvolsurface as vsurf
from sdevpy.utilities import timegrids
from sdevpy.utilities.tools import isequal
from sdevpy.maths.metrics import rmse
from sdevpy.maths import constants
from sdevpy.volatility.impliedvol.impliedvol_calib import TsIvCalibrator


class TsSvi2(ParametricImpliedVol):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_params = 15
        self.calculable_at_zero = False
        self.tmax = kwargs.get('tmax', 42)

    def calculate(self, t: float, k: npt.ArrayLike, is_call: bool, f: float) -> npt.ArrayLike:
        """ Calculate the smile parameters at given time and then calculate the Black implied vol
            using the SVI formula """
        # Check parameters
        is_ok, _ = self.check_global_params()
        if not is_ok:
            return np.full_like(k, np.nan)

        # Get smile parameters
        smile_params = self.smile_parameters(t, self.params)

        # Calculate implied vol
        log_m = np.log(k / f) # log-moneyness
        vol = svi.svi_formula(t, log_m, smile_params)
        return vol

    def smile_parameters(self, t: npt.ArrayLike, params: list[float]) -> list[float]:
        """ Calculate smile parameters according to the TsSvi2 formulas """
        if np.any(t < self.eps):
            raise ValueError("TsSvi2 is not calculable at t = 0")

        # Get parameters
        (v0, vinf, chi, tau_v, alpha, beta, rho0, rhoinf, tau_rho, m0, minf,
         tau_m, s0, sinf, tau_s) = self.get_parameters(params)
        if rho0 < -1.0 or rho0 > 1.0:
            raise ValueError("rho0 should be between -1 and 1 in TsSvi2")

        if rhoinf < -1.0 or rhoinf > 1.0:
            raise ValueError("rhoinf should be between -1 and 1 in TsSvi2")

        # Calculate wstar as integral of Rebonato function
        f0 = v0 * v0
        finf = vinf * vinf
        wstar = (finf - chi * tau_v) * t + tau_v * (chi * (t + tau_v) + f0 - finf) * (1.0 - np.exp(-t / tau_v))

        # Go back to standard parameters
        b = alpha * np.power(t, beta)
        s = s0 + (sinf - s0) * (1.0 - np.exp(-t / tau_s))
        r = rho0 + (rhoinf - rho0) * (1.0 - np.exp(-t / tau_rho))
        a = wstar - b * s * np.sqrt(1.0 - r * r)
        m = m0 + (minf - m0) * (1.0 - np.exp(-t / tau_m))

        return [a, b, r, m, s]

    def get_parameters(self, x: list[float]) -> tuple[float, ...]:
        """ Return named parameters from input list """
        if len(x) != 15:
            raise ValueError("The number of parameters should be 15 for TsSvi2")

        return x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14]

    def check_params(self) -> tuple[bool, float]:
        """ Check validity of the parameters """
        # Check global parameters
        is_ok, penalty = self.check_global_params()
        # is_ok, penalty = True, 0.0 # Skip global parameter check

        # Check local parameters over sampled expiries
        if is_ok:
            sample_times = np.asarray([1 / 365, 7 / 365, 30 / 365, 0.5, 1, 5, 10, 40])
            sample_params = self.smile_parameters(sample_times, self.params)
            is_ok, penalty = svi.svi_check_params(sample_params)

        return is_ok, penalty

    def check_global_params(self) -> tuple[bool, float]:
        """ Check validity of the global parameters """
        # Get parameters
        (v0, vinf, chi, tau_v, alpha, beta, rho0, rhoinf, tau_rho, m0, minf,
         tau_m, s0, sinf, tau_s) = self.get_parameters(self.params)
        if rho0 < -1.0 or rho0 > 1.0:
            return False, constants.FLOAT_INFTY
            # raise ValueError("rho0 should be between -1 and 1 in TsSvi2")

        if rhoinf < -1.0 or rhoinf > 1.0:
            return False, constants.FLOAT_INFTY
            # raise ValueError("rhoinf should be between -1 and 1 in TsSvi2")

        is_ok = True
        # Check necessary no-arbitrage
        no_arb1 = alpha * np.power(self.tmax, beta)
        no_arb2 = 4.0 / (1.0 + np.maximum(np.abs(rho0), np.abs(rhoinf)))
        if no_arb1 > no_arb2:
            is_ok = False

        # Check bound for extremum of vstar
        tol = 1e-6
        if is_ok and not isequal(chi, tol):
            if tau_v < tol:
                is_ok = False
            else:
                if chi < 0.0:
                    prod = chi * tau_v
                    f0 = v0 * v0
                    finf = vinf * vinf
                    v_diff = f0 - finf
                    fext = finf + prod * np.exp(-1.0 + v_diff / prod)
                    if fext < tol:
                        is_ok = False

        return is_ok, (0.0 if is_ok else constants.FLOAT_INFTY)

    def bounds(self, keep_feasible: bool=False):
        """ Recommended bounds """
        # v0, vInf, B, tauV, alpha, beta, rho0, rhoInf, tauRho, m0, mInf, tauM, sigma0, sigmaInf, tauSigma
        lw_bounds = [0.0, 0.00001, -1.0,  0.1, 0.0, 0.0001, -0.99, -0.99,  0.1, -0.99, -0.99,  0.1, 0.0, 0.0,  0.1]
        up_bounds = [1.0, 1.00000,  1.0, 50.0, 5.0, 0.9990,  0.99,  0.99, 50.0,  1.00,  1.00, 50.0, 2.0, 2.0, 50.0]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=keep_feasible)
        return bounds

    def initial_point(self) -> list[float]:
        """ Recommended initial point """
        # v0, vInf, B, tauV, alpha, beta, rho0, rhoInf, tauRho, m0, mInf, tauM, sigma0, sigmaInf, tauSigma
        # init_point = [0.10, 0.20, -0.05,  1.0, 0.1, 0.1000, -0.30, -0.30,  1.0,  1.00,  0.50,  0.5, 0.1, 0.1,  1.0]
        init_point = [0.275,  0.274, -0.0003,  0.7, 0.07,  0.003, -0.3, -0.2, 1.3,  0.001, -0.017,  0.23, 0.000001,
                      2.0, 5.0]
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
    model = TsSvi2()
    # model.update_params(model.initial_point())
    # model.check_params()

    # Calibrate model
    calibrator = TsIvCalibrator(model, {'optimizer': 'SLSQP', 'tol': 1e-6})
    calibrator.calibrate(mkt_data)
    print(f"Optimum parameters: {model.params}")

    # Estimate model on points and calculate RMSE, plot comparison
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            exp_idx = n_cols * i + j
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
