""" Step 4 draft: least-squares problem/solver layer over SciPy least_squares and scalar minimizers """
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from sdevpy.maths.optimization import SCIPY_OPTIMIZERS, SciPyOptimizer


ResidualsJac = Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
Constraint = opt.LinearConstraint | opt.NonlinearConstraint

LSQ_METHODS = ['trf', 'dogbox', 'lm']
GRADIENT_METHODS = {'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr'}
CONSTRAINED_METHODS = {'COBYLA', 'SLSQP', 'trust-constr', 'DE'}


@dataclass
class LeastSquaresProblem:
    """ min 0.5 |r(x)|^2 subject to bounds and constraints. fun returns residuals (m,) and Jacobian (m, n) """
    fun: ResidualsJac
    x0: npt.NDArray[np.float64]
    bounds: opt.Bounds | None = None
    constraints: Sequence[Constraint] = ()


@dataclass
class FitResult:
    x: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    jac: npt.NDArray[np.float64]
    cost: float  # 0.5 |r|^2
    nit: int | None  # iterations, when the backend reports them
    nfev: int  # evaluations of problem.fun
    success: bool
    message: str
    elapsed: float  # seconds
    raw: Any = None  # backend's own result


class _Cached:
    """ Evaluates fun once per distinct x: backends ask for residuals and Jacobian in separate calls """
    def __init__(self, fun: ResidualsJac):
        self.fun, self.x, self.r, self.J, self.nfev = fun, None, None, None, 0

    def __call__(self, x):
        if self.x is None or not np.array_equal(x, self.x):
            self.r, self.J = self.fun(x)
            self.x = np.array(x, copy=True)
            self.nfev += 1
        return self.r, self.J

    def residuals(self, x):
        return self(x)[0]

    def jac(self, x):
        return self(x)[1]

    def cost(self, x):
        r = self(x)[0]
        return 0.5 * float(r @ r)

    def grad(self, x):
        r, J = self(x)
        return J.T @ r


class LeastSquaresSolver(ABC):
    def solve(self, problem: LeastSquaresProblem) -> FitResult:
        cached = _Cached(problem.fun)
        t0 = time.perf_counter()
        raw, nit, success, message = self._solve(problem, cached)
        elapsed = time.perf_counter() - t0
        r, J = cached(raw.x)
        return FitResult(raw.x, r, J, 0.5 * float(r @ r), nit, cached.nfev, success, message, elapsed, raw)

    @abstractmethod
    def _solve(self, problem: LeastSquaresProblem, cached: _Cached) -> tuple[Any, int | None, bool, str]:
        """ Run the backend: (raw result with .x, iterations, success, message) """


class ScipyLeastSquaresSolver(LeastSquaresSolver):
    """ scipy.optimize.least_squares: trf (bounds), dogbox (bounds), lm (no bounds, m >= n).
        kwargs passed through: ftol, xtol, gtol, max_nfev, x_scale, loss, f_scale, ... """
    def __init__(self, method: str = 'trf', **kwargs):
        if method not in LSQ_METHODS:
            raise ValueError(f"Unknown least_squares method: {method}")
        self.method, self.kwargs = method, kwargs

    def _solve(self, problem, cached):
        if problem.constraints:
            raise ValueError(f"least_squares ({self.method}) does not support general constraints")
        bounds = (-np.inf, np.inf)
        if problem.bounds is not None:
            if self.method == 'lm':
                raise ValueError("least_squares (lm) does not support bounds")
            bounds = (problem.bounds.lb, problem.bounds.ub)
        res = opt.least_squares(cached.residuals, problem.x0, jac=cached.jac, bounds=bounds, method=self.method,
                                **self.kwargs)
        return res, None, bool(res.success), str(res.message)


class ScalarLeastSquaresSolver(LeastSquaresSolver):
    """ Any SciPyOptimizer method on f = 0.5 |r|^2, with gradient J^T r for gradient-based methods.
        The route for general constraints (SLSQP, trust-constr, COBYLA).
        x_scale: the optimizer works on u = x * scale. 'jac' (default): scale = column norms of J at x0,
        so that the Hessian of f is ~ identity in u (quasi-Newton methods start from an identity Hessian).
        Array: given scale. None: no scaling.
        options: extra method-specific SciPy options, e.g. {'initial_barrier_parameter': 1e-4} (trust-constr) """
    def __init__(self, method: str = 'SLSQP', x_scale: str | npt.ArrayLike | None = 'jac',
                 options: dict | None = None, **kwargs):
        self.method, self.x_scale = method, x_scale
        self.optimizer = SciPyOptimizer(method, **kwargs)
        self.optimizer.options.update(options or {})

    def _solve(self, problem, cached):
        if problem.constraints and self.method not in CONSTRAINED_METHODS:
            raise ValueError(f"Method {self.method} does not support general constraints")
        s = self._scale(problem, cached)
        bounds = None if problem.bounds is None else opt.Bounds(problem.bounds.lb * s, problem.bounds.ub * s)
        constraints = [_scaled_constraint(c, s) for c in problem.constraints]
        jac = (lambda u: cached.grad(u / s) / s) if self.method in GRADIENT_METHODS else None
        res = self.optimizer.minimize(lambda u: cached.cost(u / s), problem.x0 * s, bounds=bounds, jac=jac,
                                      constraints=constraints)
        res.x = res.x / s
        return res, res.get('nit'), bool(res.success), str(res.get('message', ''))

    def _scale(self, problem, cached) -> npt.NDArray[np.float64]:
        n = len(problem.x0)
        if self.x_scale is None:
            return np.ones(n)
        if isinstance(self.x_scale, str):
            if self.x_scale != 'jac':
                raise ValueError(f"Unknown x_scale: {self.x_scale}")
            norms = np.linalg.norm(cached.jac(problem.x0), axis=0)
            return np.where(norms > 0, norms, 1.0)
        return np.broadcast_to(np.asarray(self.x_scale, dtype=float), (n,))


def _scaled_constraint(c: Constraint, s: npt.NDArray[np.float64]) -> Constraint:
    """ Constraint on x expressed on u = x * s """
    if isinstance(c, opt.LinearConstraint):
        return opt.LinearConstraint(np.atleast_2d(c.A) / s, c.lb, c.ub, keep_feasible=c.keep_feasible)
    jac = c.jac if not callable(c.jac) else (lambda u: np.atleast_2d(c.jac(u / s)) / s)
    return opt.NonlinearConstraint(lambda u: c.fun(u / s), c.lb, c.ub, jac=jac, keep_feasible=c.keep_feasible)


def create_lsq_solver(method: str, **kwargs) -> LeastSquaresSolver:
    """ Least-squares solver by name: trf/dogbox/lm (least_squares) or any scalar SciPy method """
    if method in LSQ_METHODS:
        return ScipyLeastSquaresSolver(method, **kwargs)
    if method in SCIPY_OPTIMIZERS:
        return ScalarLeastSquaresSolver(method, **kwargs)
    raise ValueError(f"Least-squares solver not supported: {method}")


################## Proposed SciPyOptimizer.minimize (optimization.py) #############################
# def _patched_minimize(self, func, x0=None, args=(), bounds=None, jac=None, constraints=()):
#     result = None
#     if self.method_ in self.std_minimizers:
#         tol = self.kwargs.get('tol', None)
#         ftol = self.kwargs.get('ftol', None)
#         effective_tol = ftol if ftol is not None else tol # ftol takes priority
#         result = opt.minimize(func, x0, args, method=self.method_, jac=jac, bounds=bounds,
#                               constraints=constraints, tol=effective_tol, options=self.options)
#     elif self.method_== 'DE':
#         atol = self.kwargs.get('atol', 0)
#         popsize = self.kwargs.get('popsize', 15)
#         strategy = self.kwargs.get('strategy', 'best1bin')
#         recombination = self.kwargs.get('recombination', 0.7)
#         tol = self.kwargs.get('tol', None)
#         if tol is None:
#             tol = 0.0
#         maxiter = self.kwargs.get('maxiter', 1000)
#         result = opt.differential_evolution(func, x0=x0, args=args, bounds=bounds,
#                                             tol=tol, maxiter=maxiter, atol=atol,
#                                             popsize=popsize, strategy=strategy,
#                                             recombination=recombination, constraints=constraints)
#     else:
#         raise ValueError(f"Method {self.method_} not recognized")

#     return result


# SciPyOptimizer.minimize = _patched_minimize  # scratch only: stands in for the library change


# if __name__ == "__main__":
#     import datetime as dt
#     from sdevpy.utilities.scalendar import BDC
#     from sdevpy.utilities.sdaycount import DayCount
#     from sdevpy.utilities import timegrids
#     from sdevpy.utilities.schedules import LegConvention, make_bond_scheduler
#     from sdevpy.rates.bonds import bond_flows, flatten, prices_jac, prices_from_discounts
#     from sdevpy.rates.bondquotes import bill_flows, settlement_date
#     from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
#     from sdevpy.calibration.rates.curvebasis import CurveBasis

#     # Synthetic set: 380 bills/notes/bonds priced on a known curve, fit from a flat 4% start
#     valdate = dt.datetime(2026, 10, 26)
#     settle = settlement_date(valdate, 'USD')
#     ust = LegConvention('6M', DayCount.ACTACT_ICMA, 'USD', bdc=BDC.F, adjust_accrual=False, eom=True)
#     rng = np.random.default_rng(42)
#     flows, coupons = [], []
#     for k in range(26):
#         flows.append(bill_flows(settle + dt.timedelta(days=7 * (k + 1)), settle, valdate))
#         coupons.append(0.0)
#     for m in range(6, 360):
#         mat = dt.date(2027 + (m - 6) // 12, 1 + (m - 6) % 12, 15)
#         issue = dt.date(mat.year - 30 if m > 120 else mat.year - 10, mat.month, mat.day)
#         flows.append(bond_flows(make_bond_scheduler(issue, mat, ust), settle, valdate))
#         coupons.append(round(rng.uniform(0.01, 0.05) * 800) / 800)
#     ff = flatten(flows, coupons)
#     tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30.5]
#     node_times = timegrids.model_time(valdate, [valdate + dt.timedelta(days=round(365 * x)) for x in tenors])
#     true_dfs = np.exp(-(0.035 + 0.01 * (1 - np.exp(-node_times / 5))) * node_times)
#     w = 1.0 / (np.array([1e-4 * max(t, 0.1) for t in ff.times[ff.starts + np.diff(np.append(ff.starts, ff.n_flows)) - 1]]))
#     print(f"{ff.n_bonds} instruments, {len(tenors)} nodes")

#     for var, itype in [('zerorate', 'cubicspline'), ('logdiscount', 'cubicspline-linear')]:
#         curve = InterpolatedYieldCurve(valdate=valdate, interp_var=var, interp_type=itype)
#         cb = CurveBasis(curve, node_times, ff.times)
#         y_true = curve.node_values(node_times, true_dfs)
#         target = prices_from_discounts(ff, cb.discount(y_true))
#         y0 = curve.node_values(node_times, np.exp(-0.04 * node_times))

#         def fun(y):
#             p, jac = prices_jac(ff, *cb.discount_jac(y))
#             return w * (p - target), w[:, None] * jac

#         print(f"--- {var} {itype}")
#         for method, kw in [('trf', {}), ('trf', dict(xtol=1e-14, ftol=1e-14, gtol=1e-14)), ('dogbox', {}),
#                            ('lm', {}), ('SLSQP', dict(ftol=1e-14)), ('trust-constr', dict(gtol=1e-12)),
#                            ('L-BFGS-B', dict(ftol=1e-15, gtol=1e-12)), ('BFGS', dict(gtol=1e-12))]:
#             res = create_lsq_solver(method, **kw).solve(LeastSquaresProblem(fun, y0))
#             print(f"  {method:<12} {str(kw)[:40]:<42} {1e3 * res.elapsed:7.1f} ms  nfev {res.nfev:4d}  "
#                   f"max resid {np.max(np.abs(res.residuals)):.1e} bp  max node err {np.max(np.abs(res.x - y_true)):.1e}")
