from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
from sdevpy.calibration.rates.curvebasis import CurveBasis


ResidualsJac = Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]


class ForwardGrid:
    """ Discrete forwards F_i = -(ln DF(t_i+1) - ln DF(t_i)) / h on a uniform grid 0 = t_0 < ... < t_G = horizon,
        as functions of the node values y, with their Jacobian """
    def __init__(self, curve: InterpolatedYieldCurve, node_times: npt.ArrayLike, horizon: float, step: float):
        n_steps = max(int(np.ceil(horizon / step)), 1)
        self.grid = np.linspace(0.0, horizon, n_steps + 1)
        self.h = self.grid[1] - self.grid[0]
        self.basis = CurveBasis(curve, node_times, self.grid)

    @property
    def mid_times(self) -> npt.NDArray[np.float64]:
        """ Centre of each forward period """
        return 0.5 * (self.grid[1:] + self.grid[:-1])

    def forwards(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return -np.diff(np.log(self.basis.discount(y))) / self.h

    def forwards_jac(self, y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ F (G,) and dF/dy (G, n_nodes) """
        d, g, dv = self.basis.discount_jac(y)
        dlog = (g / d)[:, None] * dv
        return -np.diff(np.log(d)) / self.h, -np.diff(dlog, axis=0) / self.h


@dataclass(frozen=True)
class SmoothnessPenalty:
    """ Roughness of the forward curve as extra residual rows, in bp: sqrt(lam(t) h) * 1e4 * D^order F, so that
        their sum of squares ~ lam 1e8 integral (f^(order))^2 dt. order 1 penalises forward slope, order 2 forward
        curvature. lam: constant or function of maturity (larger at the long end: variable roughness penalty).
        step: grid spacing in years. horizon: grid end, default the last node """
    lam: float | Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
    order: int = 2
    step: float = 1.0 / 12.0
    horizon: float | None = None

    def build(self, curve: InterpolatedYieldCurve, node_times: npt.NDArray[np.float64]) -> ResidualsJac:
        if self.order not in (1, 2):
            raise ValueError(f"Unsupported smoothness order: {self.order}")
        fg = ForwardGrid(curve, node_times, self.horizon or node_times[-1], self.step)
        t = fg.mid_times[1:len(fg.mid_times) - self.order + 1]  # location of each difference
        lam = self.lam(t) if callable(self.lam) else np.full(len(t), float(self.lam))
        scale = 1e4 * np.sqrt(lam * fg.h) / fg.h ** self.order

        def residuals_jac(y):
            f, df = fg.forwards_jac(y)
            return scale * np.diff(f, n=self.order), scale[:, None] * np.diff(df, n=self.order, axis=0)
        return residuals_jac


@dataclass(frozen=True)
class ForwardFloor:
    """ Discrete forwards on the grid >= floor (needs a constrained solver: SLSQP, trust-constr) """
    floor: float = 0.0
    step: float = 1.0 / 12.0
    horizon: float | None = None

    def build(self, curve: InterpolatedYieldCurve, node_times: npt.NDArray[np.float64]) -> opt.NonlinearConstraint:
        fg = ForwardGrid(curve, node_times, self.horizon or node_times[-1], self.step)
        return opt.NonlinearConstraint(fg.forwards, self.floor, np.inf, jac=lambda y: fg.forwards_jac(y)[1])
