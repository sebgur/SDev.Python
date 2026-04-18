""" Term-structure model for mixtures of log-normal distributions.
    Give each distribution and weight parameters a parametric formula along time and
    enforce no-arbitrage (exactly). This model has 11 parameters.
    See D. Bloch, 'A Practical Guide to Implied and Local Volatility', 2010
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1538808
"""
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import datetime as dt
from scipy.stats import norm
import scipy.optimize as opt
from sdevpy.volatility.impliedvol.zerosurface import ParametricZeroSurface
from sdevpy.volatility.impliedvol.optionsurface import calibration_targets, IS_CALL
from sdevpy.market import eqvolsurface as vsurf
from sdevpy.tools import timegrids
from sdevpy.tools.utils import isequal
from sdevpy.maths.metrics import rmse
from sdevpy.maths import constants
from sdevpy.volatility.impliedvol.impliedvol_calib import TsIvCalibrator


class TimeParam(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def value(self, t: float) -> float:
        """ Value of the parameter at a given time """
        pass

    @abstractmethod
    def diff(self, t: float) -> float:
        """ Differential of the parameter function along time """
        pass


def logmix_f(t: float, beta: float) -> float:
    """ LogMix's intermediate function """
    tmp = 1.0 + t / beta
    return 1.0 - 2.0 / (1.0 + tmp * tmp)


def logmix_df(t: float, beta: float) -> float:
    """ LogMix's intermediate function differential """
    tmp1 = 1.0 + t / beta
    tmp2 = 1.0 + tmp1 * tmp1
    return (4.0 * tmp1 / beta) / (tmp2 * tmp2)


class LogMixMean(TimeParam):
    """ Mean (shift) parameter of the LogMix model """
    def __init__(self, mu0: float, beta: float):
        super().__init__()
        self.mu0, self.beta = mu0, beta

    def value(self, t: float) -> float:
        """ Value of the LogMixMean parameter """
        return self.mu0 * logmix_f(t, self.beta)

    def diff(self, t: float) -> float:
        """ Differential of the LogMixMean parameter """
        return self.mu0 * logmix_df(t, self.beta)


class LogMixStrike(TimeParam):
    """ Strike (shift) parameter of the LogMix model """
    def __init__(self, nu0: float, beta: float):
        super().__init__()
        self.nu0, self.beta = nu0, beta

    def value(self, t: float) -> float:
        """ Value of the LogMixStrike parameter """
        return self.nu0 * logmix_f(t, self.beta)

    def diff(self, t: float) -> float:
        """ Differential of the LogMixStrike parameter """
        return self.nu0 * logmix_df(t, self.beta)


class LogMixVar(TimeParam):
    """ Variance parameter of the LogMix model """
    def __init__(self, a: float, b: float, c: float, d: float):
        super().__init__()
        self.a, self.b, self.c, self.d = a, b, c, d

    def value(self, t: float) -> float:
        """ Value of the LogMixVar parameter """
        s = self.a * np.exp(-self.c * t) + self.d * logmix_f(t, self.b)
        return s * s * t

    def diff(self, t: float) -> float:
        """ Differential of the LogMixVar parameter """
        tmp = self.a * np.exp(-self.c * t)
        s = tmp + self.d * logmix_f(t, self.b)
        ds = -self.c * tmp + self.d * logmix_df(t, self.b)
        return s * (2.0 * ds * t + s)


class LogMixWeight(TimeParam):
    """ Weight parameter of the LogMix model """
    def __init__(self, component: int, w0: list[float], beta: list[float]):
        super().__init__()
        self.component, self.w0, self.beta = component, w0, beta
        self.n_components = len(self.w0)
        if len(self.beta) != self.n_components:
            raise ValueError("Incompatible sizes between weight and beta")

        self.norm = LogMixNorm(w0, beta)

    def value(self, t: float) -> float:
        """ Value of the LogMixWeight parameter """
        w = self.w0[self.component] / (logmix_f(t, self.beta[self.component]) * self.norm.value(t))
        return w

    def diff(self, t: float) -> float:
        """ Differential of the LogMixWeight parameter """
        tmp1 = logmix_f(t, self.beta[self.component])
        tmp2 = self.norm.value(t)
        tmp3 = tmp1 * tmp2
        w = self.w0[self.component]
        return -w / (tmp3 * tmp3) * (tmp1 * self.norm.diff(t) + logmix_df(t, self.beta[self.component]) * tmp2)


class LogMixNorm(TimeParam):
    """ Normalization parameter of the LogMix model """
    def __init__(self, w0: list[float], beta: list[float]):
        super().__init__()
        self.w0, self.beta = w0, beta
        self.n_components = len(self.w0)
        if len(self.beta) != self.n_components:
            raise ValueError("Incompatible sizes between weight and beta")

    def value(self, t: float) -> float:
        """ Value of the LogMixNorm parameter """
        norm_ = 0.0
        for i in range(self.n_components):
            norm_ += self.w0[i] / logmix_f(t, self.beta[i])

        return norm_

    def diff(self, t: float) -> float:
        """ Differential of the LogMixNorm parameter """
        dnorm = 0.0
        for i in range(self.n_components):
            tmp = logmix_f(t, self.beta[i])
            dnorm += -self.w0[i] / (tmp * tmp) * logmix_df(t, self.beta[i])

        return dnorm


class LogMix(ParametricZeroSurface):
    def __init__(self, n_mix=2, **kwargs):
        super().__init__()
        self.n_mix = kwargs.get('n_mix', 2)
        self.check_fwd_var = kwargs.get('check_fwd_var', False)
        self.shift_mean = kwargs.get('shift_mean', True)
        self.calculable_at_zero = False
        self.n_params = 5 + 7 * (self.n_mix - 1)

    def calculate(self, t: float, k: npt.ArrayLike, is_call: bool, f: float) -> npt.ArrayLike:
        """ Calculate Black price (forward) """
        return self.price(t, k, is_call, f)

    def price(self, t: float, strike: float, is_call: bool, fwd: float) -> float:
        """ Option price: weighted sum of Black-Scholes price in each component """
        price = 0.0
        for i in range(self.n_mix):
            w = self.weight[i].value(t)
            f = fwd * (1.0 + self.mean[i].value(t))
            k = strike * (1.0 + self.strike[i].value(t))
            stdev = np.sqrt(self.var[i].value(t))
            vol = stdev / np.sqrt(t)
            print(vol)
            price += w * self.black(k, is_call, f, stdev)

        return price

    def pdf(self, t: float, strike: float, fwd: float) -> float:
        """ Probability density: weighted sum of lognormal densities """
        if t < 0.0 or isequal(t, 0.0):
            raise ValueError("LogMix model cannot calculate PDF at t=0")

        prob = 0.0
        for i in range(self.n_mix):
            w = self.weight[i].value(t)
            stdev = np.sqrt(self.var[i].value(t))
            mu = 1.0 + self.mean[i].value(t)
            nu = 1.0 + self.strike[i].value(t)
            d_minus = np.log(fwd * mu / strike / nu) / stdev - 0.5 * stdev
            delta_n_minus = np.exp(-0.5 * d_minus * d_minus) / constants.C_SQRT2PI
            prob += w * delta_n_minus / stdev * nu

        return prob / strike

    def cdf(self, t: float, strike: float, fwd: float) -> float:
        """ Cumulative probability density: weighted sum of lognormal densities """
        if t < 0.0 or isequal(t, 0.0):
            raise ValueError("LogMix model cannot calculate PDF at t=0")

        prob = 0.0
        for i in range(self.n_mix):
            w = self.weight[i].value(t)
            stdev = np.sqrt(self.var[i].value(t))
            mu = 1.0 + self.mean[i].value(t)
            nu = 1.0 + self.strike[i].value(t)
            d_minus = np.log(fwd * mu / strike / nu) / stdev - 0.5 * stdev
            prob += w * norm.cdf(-d_minus) * nu

        return prob

    def black(self, strike: float, is_call: bool, fwd: float, stdev: float) -> float:
        """ Quick version to avoid calculating the vol for nothing """
        w = 1.0 if is_call else -1.0
        d1 = np.log(fwd / strike) / stdev + 0.5 * stdev
        d2 = d1 - stdev
        return w * (fwd * norm.cdf(w * d1) - strike * norm.cdf(w * d2))

    def update_params(self, x: list[float]) -> None:
        """ Update the current parameters """
        self.params = x
        self.set_param_functions(self.params)

    def set_param_functions(self, params: list[float]) -> None:
        """ Given the parameters as a list, set the parameter functions """
        is_valid, param_dic = get_logmix_parameters(self.n_mix, params)
        if not is_valid:
            raise ValueError("Invalid parameter list for LogMix model")

        # Get parameter vectors
        w, shift, beta = param_dic['w'], param_dic['shift'], param_dic['beta']
        a, b, c, d = param_dic['a'], param_dic['b'], param_dic['c'], param_dic['d']

        self.weight, self.mean, self.strike, self.var = [], [], [], []
        for i in range(self.n_mix):
            self.weight.append(LogMixWeight(i, w, beta))
            self.var.append(LogMixVar(a[i], b[i], c[i], d[i]))
            if self.shift_mean:
                self.mean.append(LogMixMean(shift[i], beta[i]))
                self.strike.append(LogMixStrike(0.0, 1.0))
            else:
                self.mean.append(LogMixMean(0.0, 1.0))
                self.strike.append(LogMixStrike(shift[i], beta[i]))

    def check_params(self) -> tuple[bool, float]:
        """ Check validity of the parameters """
        is_ok, param_dic = get_logmix_parameters(self.n_mix, self.params)

        if is_ok: # Check further constraints on the params
            w, shift, _ = param_dic['w'], param_dic['shift'], param_dic['beta']
            a, b, c, d = param_dic['a'], param_dic['b'], param_dic['c'], param_dic['d']

            eps = 0.00000001
            # Positivity of the first weight
            if is_ok and w[0] < eps:
                is_ok = False

            # Positivity of the first shift
            if is_ok and shift[0] < -1.0 + eps:
                is_ok = False

            # Positivity of the forward variances
            if is_ok and self.check_fwd_var:
                n_points = 40
                n_ = float(n_points)
                max_t = 20.0
                for i in range(self.n_mix):
                    if not is_ok:
                        break

                    var_param = LogMixVar(a[i], b[i], c[i], d[i])
                    for j in range(n_points):
                        if not is_ok:
                            break

                        fwd_v = var_param.diff(max_t * float(j) / n_)
                        is_ok = (fwd_v > 0.0)

        return is_ok, (0.0 if is_ok else constants.FLOAT_INFTY)

    def bounds(self, keep_feasible: bool=False):
        """ Recommended bounds """
        raise NotImplementedError("bounds")

    def initial_point(self) -> list[float]:
        """ Recommended initial point """
        w0, shift0, beta0 = 0.0, 0.0, 0.2
        a0, b0, c0, d0 = 0.2, 0.2, 1.0, 1.0

        init_params = [beta0, a0, b0, c0, d0]
        for _ in range(1, self.n_mix):
            init_params.append(w0)
            init_params.append(shift0)
            init_params.append(beta0)
            init_params.append(a0)
            init_params.append(b0)
            init_params.append(c0)
            init_params.append(d0)

        return init_params

def get_logmix_parameters(n_mix: int, params: npt.ArrayLike):
    """ Given the parameters as a list and knowing n_mix (i.e. number of lognormal components),
        strip the LogMix parameters out """
    loc0_thresh = 0.00000001
    # Initialize parameters vectors
    w = np.empty(n_mix)
    shift = np.empty(n_mix)
    beta = np.empty(n_mix)
    a = np.empty(n_mix)
    b = np.empty(n_mix)
    c = np.empty(n_mix)
    d = np.empty(n_mix)

    if len(params) % 7 != 5 or n_mix < 1:
        raise ValueError("Inconsistent parameter sizes in LogMix parameters")

    beta[0] = params[0]
    a[0] = params[1]
    b[0] = params[2]
    c[0] = params[3]
    d[0] = params[4]
    tmp_w = 1.0
    tmp_n = 0.0
    for i in range(1, n_mix):
        w_ = params[7 * i - 2]
        n_ = params[7 * i - 1]
        w[i] = w_
        shift[i] = n_
        beta[i] = params[7 * i]
        a[i] = params[7 * i + 1]
        b[i] = params[7 * i + 2]
        c[i] = params[7 * i + 3]
        d[i] = params[7 * i + 4]
        tmp_w -= w_
        tmp_n -= w_ * n_

    w[0] = tmp_w
    shift[0] = tmp_n / tmp_w

    param_dic = {'w': w, 'shift': shift, 'beta': beta, 'a': a, 'b': b, 'c': c, 'd': d}
    is_valid = (tmp_w >= loc0_thresh)
    return is_valid, param_dic


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

    # Set calibration time grid
    expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

    # Set calibration targets
    cf_price_surface, ftols = calibration_targets(expiry_grid, fwds, strike_surface, vol_surface)

    # Initialize model
    model = LogMix(n_mix=2)
    model.update_params(model.initial_point())
    print(model.check_params())

    # # Calibrate model
    # calibrator = TsIvCalibrator(model, {'optimizer': 'SLSQP', 'tol': 1e-10})
    # calibrator.calibrate(mkt_data)

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
            ax.scatter(strikes, cf_price_surface[exp_idx], label="market", color='black')
            # ax.scatter(strikes, vol_surface[exp_idx], label="market", color='black')
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
