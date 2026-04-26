from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.optimize import brentq
from sdevpy.maths import constants
from sdevpy.analytics import black, bachelier
from sdevpy.volatility.impliedvol.optionsurface import OptionQuoteType


class LvMethod(Enum):
    ImpliedVol = 0
    PDF = 1


class ImpliedVol(ABC):
    def __init__(self):
        self.calculate_type = OptionQuoteType.LogNormalVol
        self.shift = 0.0 # In Math format, i.e. 0.01 for 1%
        self.allow_negative_variables = False
        self.calculable_at_zero = True
        self.lv_method = LvMethod.ImpliedVol
        self.daycount = None
        self.expiry_times = []
        self.eps = constants.EPS
        self.time_epsilon = 0.000001
        self.base_date = None

    @abstractmethod
    def calculate(self, t: float, k: npt.ArrayLike, is_call: bool, f: float) -> npt.ArrayLike:
        pass

    ############### Dupire Logic ##################################################################

    def volatility(self, t: float, x: npt.ArrayLike) -> npt.ArrayLike:
        """ Black volatility for time t and moneyness x """
        return self.black_volatility(t, x, 1.0)

    def dvariance_dt(self, ts: float, te: float, x: float) -> float:
        """ Differential of variance against time between times ts and te, at moneyness x """
        tmpe = self.volatility(te, x)
        tmps = self.volatility(ts, x)
        return (tmpe * tmpe * te - tmps * tmps * ts) / (te - ts)

    def taylor_dx(self, t: float, x: npt.ArrayLike) -> npt.ArrayLike:
        """ Differential of volatility against moneyness, order 1 and 2 """
        hr = 0.05 # Relative bump
        # print(f"x-shape: {x.shape}")
        dx = hr * x
        # print(f"dx-shape: {dx.shape}")

        vol = self.volatility(t, x)
        vol_up = self.volatility(t, x + dx)
        vol_dn = self.volatility(t, x - dx)
        dvol_dx = (vol_up - vol_dn) / (2.0 * dx)
        d2vol_dx2 = (vol_up + vol_dn - 2.0 * vol) / np.power(dx, 2)
        return vol, dvol_dx, d2vol_dx2

    def dvolatility_dx(self, t: float, x: float) -> float:
        """ Differential of volatility against moneyness """
        hr = 0.05 # Relative bump
        dx = hr * x

        vol_up = self.volatility(t, x + dx)
        vol_dn = self.volatility(t, x - dx)
        dvol_dx = (vol_up - vol_dn) / (2.0 * dx)
        return dvol_dx

        ## Old formula with absolute bumps
        # h = 0.001
        # if x - h < 0.0:
        #     raise ValueError("Negative strike in numerical 1st differential of implied volatility")

        # tmp1 = self.volatility(t, x + h)
        # tmp2 = self.volatility(t, x - h)
        # return (tmp1 - tmp2) / (2.0 * h)

    def d2volatility_dx2(self, t: float, x: float) -> float:
        """ Second differential of the volatility against the moneyness """
        hr = 0.05 # Relative bump
        # print(f"x-shape: {x.shape}")
        dx = hr * x
        # print(f"dx-shape: {dx.shape}")

        vol = self.volatility(t, x)
        vol_up = self.volatility(t, x + dx)
        vol_dn = self.volatility(t, x - dx)
        # dvol_dx = (vol_up - vol_dn) / (2.0 * dx)
        d2vol_dx2 = (vol_up + vol_dn - 2.0 * vol) / np.power(dx, 2)
        return d2vol_dx2

        ## Old formula with absolute bumps
        # h = 0.001
        # two_h = 2.0 * h
        # if x - two_h < 0.0:
        #     raise ValueError("Negative strike in numerical 2nd differential of implied volatility")

        # tmp = self.volatility(t, x)
        # dxp = (self.volatility(t, x + two_h) - tmp) / two_h
        # dxm = (tmp - self.volatility(t, x - two_h)) / two_h
        # return (dxp - dxm) / two_h

    def density(self, t: float, fwd: float, strike: npt.ArrayLike) -> npt.ArrayLike:
        """ Probability density corresponding to the surface """
        if np.abs(t) < self.time_epsilon:
            raise ValueError("Probability density cannot be calculated at t = 0")

        # Get from implied volatility
        strike = np.asarray(strike)
        x = strike / fwd
        sqrt_t = np.sqrt(t)

        # Substitute safe x beore any model eval to avoid NaN at near-zero moneyness
        zero_x_mask = (x < self.eps)
        safe_x = np.where(zero_x_mask, 1.0, x) # Replace by 1.0 (atm) where we won't use it

        # Get Taylor components
        vol, dvol_dx, d2vol_dx2 = self.taylor_dx(t, safe_x)

        stdev = self.volatility(t, x) * sqrt_t

        if np.abs(stdev) < self.eps:
            raise ValueError("Probability density cannot be calculated at standard deviation 0")

        # if np.abs(stdev) < self.eps:
        #     raise ValueError("Probability density cannot be calculated at standard deviation 0")

        # if x < self.eps: # 0 or negative
        #     return 0.0

        xdtheta_dx = x * self.dvolatility_dx(t, x)
        x2d2theta_dx2 = x * x * self.d2volatility_dx2(t, x)
        d_minus = -np.log(x) / stdev - 0.5 * stdev
        d_plus_sqrt_t = (d_minus + stdev) * sqrt_t
        delta_n_minus = np.exp(-0.5 * d_minus * d_minus) / constants.C_SQRT2PI
        tmp = 1.0 + d_plus_sqrt_t * xdtheta_dx
        main = x2d2theta_dx2 - d_plus_sqrt_t * xdtheta_dx * xdtheta_dx + tmp * tmp / (stdev * sqrt_t)
        return sqrt_t * delta_n_minus * main / strike

    def cumulative(self, t: float, fwd: float, strike: float) -> float:
        """ Cumulative function of the surface's probability density """
        x = strike / fwd
        theta = self.volatility(t, x)
        sqrt_t = np.sqrt(t)
        stdev = theta * sqrt_t
        dm = -np.log(x) / stdev - 0.5 * stdev
        dtheta = self.dvolatility_dx(t, x)
        return norm.pdf(dm) * x * sqrt_t * dtheta - norm.cdf(dm) + 1.0

    def cumulative_inverse(self, t: float, p: float) -> float:
        """ Inverse cumulative function of the surface's probability density """
        if np.abs(t) < self.time_epsilon:
            raise ValueError("Cumulative inverse at t = 0 is not defined for implied volatility")

        if np.abs(p) < 1e-10:
            return 0.0

        # cumulativeFunction = CumulativeFunction(self, t)
        # solver = new ZBrent(1e-6, 100.0, 1000000, 0.000000001)
        fwd = 1.0
        result = brentq(f=lambda x: self.cumulative(t, fwd, x) - p, a=1e-6, b=100.0, xtol=1e-9, maxiter=1000)
        return result

    ############### Price-Vol Logic ###############################################################

    def to_price(self, t: float, k: float, f: float, is_call: bool, value: float) -> float:
        """ For converstion to price, given the calculated value """
        match self.calculate_type:
            case OptionQuoteType.ForwardPremium:
                return value
            case OptionQuoteType.LogNormalVol:
                return black.price(t, k, is_call, f, value)
            case OptionQuoteType.NormalVol:
                return bachelier.price(t, k, is_call, f, value)
            case OptionQuoteType.ShiftedLogNormalVol:
                return black.price(t, k + self.shift, is_call, f + self.shift, value)
            case _:
                raise TypeError(f"Invalid modelled type in zero-surface: {self.calculate_type}")

    def forward_price(self, t: float, k: float, is_call: bool, f: float) -> float:
        """ Forward price """
        value = self.calculate(t, k, is_call, f)
        return self.to_price(t, k, f, is_call, value)

    def black_volatility(self, t: float, k: float, f: float) -> float:
        """ Black implied volatility """
        is_call = True
        value = self.calculate(t, k, is_call, f)
        if self.calculate_type == OptionQuoteType.LogNormalVol:
            return value
        else:
            price = self.to_price(t, k, f, is_call, value)
            return black.implied_vols(t, k, is_call, f, price)

    def bachelier_volatility(self, t: float, k: float, f: float):
        """ Bachelier implied volatility """
        is_call = True
        value = self.calculate(t, k, is_call, f)
        if self.calculate_type == OptionQuoteType.NormalVol:
            return value
        else:
            price = self.to_price(t, k, f, is_call, value)
            return bachelier.implied_vol_jaeckel(t, k, is_call, f, price)

    def shifted_black_volatility(self, t: float, k: float, f: float) -> float:
        """ Shifted Black implied volatility """
        is_call = True
        value = self.calculate(t, k, is_call, f)
        if self.calculate_type == OptionQuoteType.ShiftedLogNormalVol:
            return value
        else:
            price = self.to_price(t, k, f, is_call, value)
            return black.implied_vol(t, k + self.shift, is_call, f + self.shift, price)


class ParametricImpliedVol(ImpliedVol):
    def __init__(self):
        super().__init__()
        self.n_params = None
        self.params = None

    def update_params(self, x: list[float]) -> None:
        """ Update the current parameters """
        self.params = x

    def check_params(self) -> tuple[bool, float]:
        """ Check validity of the parameters, return is_ok state and penalty value (0.0 if is_ok) """
        return True, 0.0

    @abstractmethod
    def bounds(self, keep_feasible: bool=False):
        """ Recommended bounds for optimization on parameters """
        pass

    @abstractmethod
    def initial_point(self) -> list[float]:
        """ Recommended initial point for optimization on parameters """
        pass
