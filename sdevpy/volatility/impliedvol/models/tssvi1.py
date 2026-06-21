""" Term-structure model for SVI. Give each SVI parameter a parametric formula along time and
    enforce no-arbitrage (approximately). This model has 11 parameters.
    See Gurrieri, 'A Class of Term Structures for SVI Implied Volatility', 2010
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1779463
    Beware of the reparameterization in the paper above, where the original SVI formula is
    applied to the squared vol instead of the variance. This appears to be rather odd.
    In the below we parameterize using the original SVI, but the idea remains nearly identical
    to that in the paper.
    However, these seems to be something strange with SVI's original paper's application of
    the Rogers-Tehranchi bound, which we fix here. Indeed Rogers-Tehranchi looks like
    it is on the total variance, so should be applied directly to SVI.
"""
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from sdevpy.volatility.impliedvol.parametric_impliedvol import ParametricImpliedVol
from sdevpy.volatility.impliedvol.models import svi
from sdevpy.utilities.tools import isequal
from sdevpy.maths import constants


class TsSvi1(ParametricImpliedVol):
    def __init__(self, **kwargs):
        super().__init__()
        self.n_params = 11
        self.calculable_at_zero = False
        self.tmax = kwargs.get('tmax', 42)

    def calculate(self, t: float, k: npt.ArrayLike, is_call: bool, f: float) -> npt.ArrayLike:
        """ Calculate the smile parameters at given time and then calculate the Black implied vol
            using the SVI formula """
        # Check initialization
        if self.params is None:
            raise RuntimeError("No parameters set in model yet: set with update_params()")

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

    def taylor_dx(self, t: float, x: npt.ArrayLike) -> npt.ArrayLike:
        """ Analytical differential of volatility against moneyness, order 1 and 2.
            If this method is commented out, we will revert to the generic method
            in the base class, going through finite differences.
        """
        # Get smile parameters
        smile_params = self.smile_parameters(t, self.params)

        # Calculate sensitivities to the log-moneyness
        log_x = np.log(x) # log-moneyness
        vol, dvol_dlog_x, d2vol_dlog_x2 = svi.taylor_dlog_x(t, log_x, smile_params)

        # Calculate sensitivities to the moneyness
        dlog_x_dx = 1.0 / x
        dvol_dx = dvol_dlog_x * dlog_x_dx
        d2vol_dx2 = d2vol_dlog_x2 * dlog_x_dx**2 - dvol_dlog_x / x**2

        return vol, dvol_dx, d2vol_dx2

    def smile_parameters(self, t: npt.ArrayLike, params: list[float]) -> list[float]:
        """ Calculate smile parameters according to the TsSvi1 formulas """
        if np.any(t < self.eps):
            raise ValueError("TsSvi1 is not calculable at t = 0")

        # Get parameters
        s0, sinf, chi, tau, alpha, beta, r, x0star, lambda0, gamma, delta = self.get_parameters(params)

        # Calculate new variables
        one_minus_rho2 = max(1.0 - r * r, 0.0)

        # Vectorize r (the other ones broadcast fine thanks to t)
        t = np.asarray(t, dtype=float)
        r = np.full_like(t, r)

        pow_ = beta + delta
        wstar = alpha * gamma * one_minus_rho2 / (pow_ + 1.0) * np.power(t, pow_ + 1.0)
        v0 = s0 * s0
        vinf = sinf * sinf
        wstar += (vinf - chi * tau) * t + tau * (chi * (t + tau) + v0 - vinf) * (1.0 - np.exp(-t / tau))

        lambda_ = lambda0 + gamma / (delta + 1.0) * np.power(t, delta + 1.0)
        xstar = x0star - r * (lambda_ - lambda0)

        # Go back to standard parameters
        b = alpha * np.power(t, beta)
        a = wstar - b * lambda_ * one_minus_rho2
        m = xstar + r * lambda_
        s = lambda_ * np.sqrt(one_minus_rho2)

        return [a, b, r, m, s]

    def get_parameters(self, x: list[float]) -> tuple[float, ...]:
        """ Return named parameters from input list """
        if len(x) != 11:
            raise ValueError("The number of parameters should be 11 for TsSvi1")

        return x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]

    def check_params(self) -> tuple[bool, float]:
        """ Check validity of the parameters """
        # Check global parameters
        is_ok, penalty = self.check_global_params()

        # Check local parameters over sampled expiries
        if is_ok:
            sample_times = np.asarray([1 / 365, 7 / 365, 30 / 365, 0.5, 1, 5, 10, 40])
            sample_params = self.smile_parameters(sample_times, self.params)
            is_ok, penalty = svi.svi_check_params(sample_params)

        return is_ok, penalty

    def check_global_params(self) -> tuple[bool, float]:
        """ Check validity of the global parameters """
        if self.params is None:
            return False, constants.FLOAT_INFTY

        # Get parameters
        s0, sinf, chi, tau, alpha, beta, r, x0star, lambda0, gamma, delta = self.get_parameters(self.params)
        if r < -1.0 or r > 1.0:
            return False, constants.FLOAT_INFTY

        if delta + 1.0 < self.eps:
            return False, constants.FLOAT_INFTY

        is_ok = True
        # Check necessary no-arbitrage
        no_arb1 = alpha * np.power(self.tmax, beta)
        no_arb2 = 4.0 / (1.0 + np.abs(r)) # Our interpretation of Rogers-Tehranchi
        if no_arb1 > no_arb2:
            is_ok = False

        # Check bound for extremum of vstar
        tol = 1e-6
        if is_ok and not isequal(chi, tol):
            if tau < tol:
                is_ok = False
            else:
                if chi < 0.0:
                    prod = chi * tau
                    vinf = sinf * sinf
                    v_diff = s0 * s0 - vinf
                    fext = vinf + prod * np.exp(-1.0 + v_diff / prod)
                    if fext < tol:
                        is_ok = False

        return is_ok, (0.0 if is_ok else constants.FLOAT_INFTY)

    def bounds(self, keep_feasible: bool=False):
        """ Recommended bounds """
        # s0, sinf, chi, tau, alpha, beta, r, x0star, lambda0, gamma, delta
        lw_bounds = [0.0, 0.00001, -1.0,  0.1, 0.00, 0.0001, -0.99, -0.50, 0.0, 0.0, -0.999]
        up_bounds = [1.0, 1.00000,  1.0, 50.0, 5.00, 0.9990,  0.20,  2.00, 2.0, 5.0,  5.000]
        bounds = opt.Bounds(lw_bounds, up_bounds, keep_feasible=keep_feasible)
        return bounds

    def initial_point(self) -> list[float]:
        """ Recommended initial point """
        # s0, sinf, chi, tau, alpha, beta, r, x0star, lambda0, gamma, delta
        init_point = [0.10, 0.20, -0.05,  1.0, 0.10, 0.10, -0.30,  0.10, 0.1, 1.0,  1.000]
        return np.asarray(init_point)

    def dump_data(self) -> dict:
        """ Dump to dictionary """
        self._require_params()
        return {'type': 'TsSvi1', 'params': self.params.tolist()}


if __name__ == "__main__":
    from scipy.stats import norm
    from sdevpy.maths import metrics
    # import matplotlib.pyplot as plt

    params = [0.2277, 0.22491, 0.000126, 1.7, 0.069, 0.0001, -0.22, 0.002,
              0.0021, 0.36, 0.012]
    model = TsSvi1()
    model.update_params(params)

    # Dupire grids
    n_points_per_year = 10
    n_strikes = 25
    lw_percent = 0.001
    up_percent = 1.0 - lw_percent
    ts, te = 0.1, 0.2

    # Create moneynesses axis
    atm_vol = model.black_volatility(te, 1.0, 1.0) # ATM
    stdev = atm_vol * np.sqrt(te)
    low_k = np.exp(-0.5 * stdev * stdev + stdev * norm.ppf(lw_percent))
    up_k = np.exp(-0.5 * stdev * stdev + stdev * norm.ppf(up_percent))
    m = np.linspace(low_k, up_k, n_strikes)

    # Dupire
    # lv_slice = dupire_formula(model, ts, te, m)


    # taylor_dx
    theta, dtheta_dx, d2theta_dx2 = model.taylor_dx(ts, m)
    # print(f"taylor_dx theta: {theta}")
    # print(f"taylor_dx dtheta_dx: {dtheta_dx}")
    # print(f"taylor_dx d2theta_dx2: {d2theta_dx2}")

    # manual taylor
    hr = 0.01 # Relative bump
    dx = hr * m
    vol = model.volatility(ts, m)
    vol_up = model.volatility(ts, m + dx)
    vol_dn = model.volatility(ts, m - dx)
    dvol_dx = (vol_up - vol_dn) / (2.0 * dx)
    d2vol_dx2 = (vol_up + vol_dn - 2.0 * vol) / np.power(dx, 2)
    # print(f"manual taylor theta: {vol}")
    # print(f"manual taylor dtheta_dx: {dvol_dx}")
    # print(f"manual taylor d2theta_dx2: {d2vol_dx2}")

    print(f"check taylor theta: {vol - theta}")
    print(f"check taylor dtheta_dx: {dvol_dx - dtheta_dx}")
    print(f"check taylor d2theta_dx2: {d2vol_dx2 - d2theta_dx2}")

    # Get parameters
    smile_params = model.smile_parameters(ts, params)
    # print(smile_params)
    a, b, r, m_, s = smile_params
    # print(a)
    # print(b)

    # Analytical sensies
    log_m = np.log(m) # log-moneyness
    cf_vol = svi.svi_formula(ts, log_m, smile_params)
    # print(f"analytical theta: {cf_vol}")
    print(f"check analytical theta: {cf_vol - theta}")

    # Calculate
    xm = log_m - m_ # x is the log-moneyness
    sqrt_ = np.sqrt(xm**2 + s**2)
    var = a + b * (r * xm + np.sqrt(xm**2 + s**2))
    cf_vol_int = np.sqrt(var / ts)
    print(f"check cf_vol_int: {cf_vol_int - theta}")

    dxm_dx = 1.0 / m

    # 1st diff
    cf_dvol_dvar = 1.0 / (2.0 * cf_vol_int) / ts
    cf_dvar_dxm = b * r + b / sqrt_ * xm
    cf_dvol_dxm = cf_dvol_dvar * cf_dvar_dxm
    cf_dvol_dx = cf_dvol_dxm * dxm_dx
    # print(f"analytical dtheta_dx: {dvol_dx}")
    # print(f"check analytical dtheta_dx: {(cf_dvol_dx - dvol_dx) / dvol_dx * 100.0}")
    # for i in range(len(dtheta_dx)):
    #     print(f"cf/fdm: {dvol_dx[i]}/{dtheta_dx[i]}")

    # 2nd diff
    cf_d2vol_dvar2 = 0.5 * (-0.5) / np.power(cf_vol_int, 3) / ts**2
    term1 = cf_d2vol_dvar2 * cf_dvar_dxm**2 * dxm_dx**2

    cf_d2var_dxm2 = b * (-0.5) / np.power(sqrt_, 3) * 2.0 * xm**2 + b / sqrt_
    term2 = cf_dvol_dvar * cf_d2var_dxm2 * dxm_dx**2

    term3 = cf_dvol_dvar * cf_dvar_dxm * (-1.0 / m**2)

    cf_d2vol_dx2 = term1 + term2 + term3

    print(f"rmse(theta): {metrics.rmse(cf_vol_int, vol) * 10000}")
    print(f"rmse(dtheta_dx): {metrics.rmse(cf_dvol_dx, dvol_dx) * 10000}")
    print(f"rmse(d2theta_dx2): {metrics.rmse(cf_d2vol_dx2, d2vol_dx2) * 10000}")

    print(f"rmse(theta): {metrics.rmse(cf_vol_int, vol) * 10000}")
    print(f"rmse(dtheta_dx): {metrics.rmse(cf_dvol_dx, dtheta_dx) * 10000}")
    print(f"rmse(d2theta_dx2): {metrics.rmse(cf_d2vol_dx2, d2theta_dx2) * 10000}")
