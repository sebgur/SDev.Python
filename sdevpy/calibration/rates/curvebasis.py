import numpy as np
import numpy.typing as npt
# import scipy.sparse as sps
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve, YieldCurveVariable
# from sdevpy.rates.bonds import FlatFlows


class CurveBasis:
    """ Curve values at fixed times as a function of the node values y (in the interpolation variable).
        Linear interpolations (linear, cubic spline, b-spline, ... with flat/linear extrapolation):
        v(t) = c(t) + B(t) y exactly, B computed once. Others (pchip, akima): through the interpolator,
        derivatives by finite differences. Linearity is detected numerically at construction. """
    def __init__(self, curve: InterpolatedYieldCurve, node_times: npt.ArrayLike, times: npt.ArrayLike,
                 fd_bump: float = 1e-8, linear_tol: float = 1e-10):
        self.curve = curve
        self.node_times = np.asarray(node_times, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.fd_bump = fd_bump
        n = len(self.node_times)

        # Basis by unit vectors, then check superposition on random node values
        self.offset = self._interp_values(np.zeros(n))
        self.basis = np.column_stack([self._interp_values(e) - self.offset for e in np.eye(n)])
        rng = np.random.default_rng(1234)
        y1, y2 = rng.uniform(0.01, 0.05, n), rng.uniform(0.01, 0.05, n)
        mix = 0.3 * self._interp_values(y1) + 0.7 * self._interp_values(y2)
        err = np.max(np.abs(self._interp_values(0.3 * y1 + 0.7 * y2) - mix))
        self.linear = bool(err < linear_tol * max(1.0, np.max(np.abs(mix))))

    @property
    def n_nodes(self) -> int:
        return len(self.node_times)

    def _interp_values(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.curve.set_nodes(self.node_times, y)
        return self.curve.interp.value(self.times)

    def values(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.offset + self.basis @ y if self.linear else self._interp_values(y)

    def values_jac(self, y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """ v (n_times,) and dv/dy (n_times, n_nodes) """
        if self.linear:
            return self.offset + self.basis @ y, self.basis
        v = self._interp_values(y)
        dv = np.empty((len(self.times), self.n_nodes))
        for k in range(self.n_nodes):
            yk = y.copy()
            yk[k] += self.fd_bump
            dv[:, k] = (self._interp_values(yk) - v) / self.fd_bump
        return v, dv

    def discount(self, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self._to_discount(self.values(y))

    def discount_jac(self, y: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                                                                npt.NDArray[np.float64]]:
        """ DF (n_times,) and its Jacobian in factored form dDF/dy = g[:, None] * dv,
            g = dDF/dv (n_times,), dv = dv/dy (n_times, n_nodes) """
        v, dv = self.values_jac(y)
        d = self._to_discount(v)
        match self.curve.interp_var:
            case YieldCurveVariable.ZERORATE:  # DF = exp(-z t)
                return d, -self.times * d, dv
            case YieldCurveVariable.DISCOUNT:
                return d, np.ones_like(d), dv
            case YieldCurveVariable.LOGDISCOUNT:  # DF = exp(l)
                return d, d, dv

    def _to_discount(self, v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        match self.curve.interp_var:
            case YieldCurveVariable.ZERORATE:
                return np.exp(-v * self.times)
            case YieldCurveVariable.DISCOUNT:
                return v
            case YieldCurveVariable.LOGDISCOUNT:
                return np.exp(v)
