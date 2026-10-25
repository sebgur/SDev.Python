""" Bond curve construction: global least-squares fit of an InterpolatedYieldCurve to bond and bill quotes """
import datetime as dt
import logging
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from sdevpy.utilities import timegrids
from sdevpy.maths.leastsquares import LeastSquaresProblem, LeastSquaresSolver, FitResult, Constraint, create_lsq_solver
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
from sdevpy.calibration.rates.curvebasis import CurveBasis
from sdevpy.rates.bondquotes import BondQuote, target_dirty
from sdevpy.rates.bonds import FlatFlows, flatten, prices_jac, yields_from_dirty, dv01s
log = logging.getLogger(__name__)


@dataclass
class BondCurveFit:
    """ Fitted curve and diagnostics. Residuals are weighted price errors, market minus model: with the default
        1/DV01 weights, approximately model minus market yield in bp """
    curve: InterpolatedYieldCurve
    result: FitResult
    quotes: list[BondQuote]
    target_prices: npt.NDArray[np.float64]  # market dirty prices
    model_prices: npt.NDArray[np.float64]  # dirty prices on the fitted curve
    weights: npt.NDArray[np.float64]
    node_times: npt.NDArray[np.float64]
    flat_flows: FlatFlows

    @property
    def residuals(self) -> npt.NDArray[np.float64]:
        return self.weights * (self.target_prices - self.model_prices)

    def yield_errors(self) -> npt.NDArray[np.float64]:
        """ Exact model minus market yield (street convention), in bp """
        ff = self.flat_flows
        return 1e4 * (yields_from_dirty(ff, self.model_prices) - yields_from_dirty(ff, self.target_prices))


def fit_bond_curve(quotes: Sequence[BondQuote], valdate: dt.datetime, node_dates: Sequence[dt.datetime],
                   interp_var: str = 'zerorate', interp_type: str = 'cubicspline',
                   solver: LeastSquaresSolver | str = 'trf', weights: str | npt.ArrayLike | None = 'dv01',
                   y0: npt.ArrayLike | None = None, initial_rate: float = 0.03, bounds: opt.Bounds | None = None,
                   constraints: Sequence[Constraint] = (), name: str = '') -> BondCurveFit:
    """ Fit the curve's node values (in its interpolation variable, at node_dates) to the quotes' dirty prices.
        quotes: flows built with this valdate. weights: 'dv01' (1/DV01: residuals ~ yield errors in bp),
        None (price errors) or one weight per quote. y0: initial node values, default a flat initial_rate
        zero curve. bounds/constraints: on the node values, passed to the solver """
    quotes = list(quotes)
    if any(q.flows.pay_times is None for q in quotes):
        raise ValueError("Quotes with flows built without valdate: no times for curve pricing")

    # Market targets and weights
    ff = flatten([q.flows for q in quotes], [q.coupon for q in quotes], [q.redemption for q in quotes])
    targets = np.array([target_dirty(q) for q in quotes])
    if isinstance(weights, str):
        if weights != 'dv01':
            raise ValueError(f"Unknown weights: {weights}")
        w = 1.0 / dv01s(ff, yields_from_dirty(ff, targets))
    elif weights is None:
        w = np.ones(len(quotes))
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=float), (len(quotes),))

    # Curve and its basis at the distinct flow and settlement times
    curve = InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=interp_type, name=name)
    node_times = np.asarray(timegrids.model_time(valdate, list(node_dates)), dtype=float)
    if np.any(np.diff(node_times) <= 0) or node_times[0] <= 0:
        raise ValueError("Node dates must be increasing and after valdate")
    cb = CurveBasis(curve, node_times, ff.times)
    if y0 is None:
        y0 = curve.node_values(node_times, np.exp(-initial_rate * node_times))

    def residuals_jac(y):
        p, jac = prices_jac(ff, *cb.discount_jac(y))
        return w * (targets - p), -w[:, None] * jac

    if isinstance(solver, str):
        solver = create_lsq_solver(solver)
    problem = LeastSquaresProblem(residuals_jac, np.asarray(y0, dtype=float), bounds, constraints)
    result = solver.solve(problem)
    if not result.success:
        log.warning(f"Bond curve fit did not converge: {result.message}")

    # Final curve through the standard path: node DFs from the fitted node values
    curve.set_nodes(node_times, result.x)
    curve.set_data(list(node_dates), curve.discount_float(node_times))
    model = targets - result.residuals / w
    return BondCurveFit(curve, result, quotes, targets, model, w, node_times, ff)


if __name__ == "__main__":
    from sdevpy.utilities.scalendar import BDC
    from sdevpy.utilities.sdaycount import DayCount
    from sdevpy.utilities.schedules import LegConvention, make_bond_scheduler
    from sdevpy.rates.bonds import bond_flows, dirty_from_curve, accrued_interest, yield_from_dirty
    from sdevpy.rates.bondquotes import (QuoteType, bill_flows, settlement_date, discount_rate_from_price)

    # Known curve: Nelson-Siegel zero rates, 3.8% short end, dip to ~3.6% at 2Y, 4.6% at 30Y
    valdate = dt.datetime(2026, 10, 26)
    settle = settlement_date(valdate, 'USD')
    def true_df(t):
        x = np.maximum(t, 1e-8) / 2.0
        f = (1.0 - np.exp(-x)) / x
        return np.exp(-(0.048 - 0.010 * f - 0.020 * (f - np.exp(-x))) * t)

    # UST-like universe: bills quoted on discount rate, notes/bonds on clean price or yield
    ust = LegConvention('6M', DayCount.ACTACT_ICMA, 'USD', bdc=BDC.F, adjust_accrual=False, eom=True)
    rng = np.random.default_rng(42)
    quotes = []
    for k in range(26):
        bf = bill_flows(settle + dt.timedelta(days=7 * (k + 1)), settle, valdate)
        p = dirty_from_curve(bf, 0.0, true_df)
        quotes.append(BondQuote(bf, 0.0, QuoteType.DISCOUNT, discount_rate_from_price(bf, p)))
    for m in range(6, 360):
        mat = dt.date(2027 + (m - 6) // 12, 1 + (m - 6) % 12, 15)
        issue = dt.date(mat.year - 30 if m > 120 else mat.year - 10, mat.month, mat.day)
        bf = bond_flows(make_bond_scheduler(issue, mat, ust), settle, valdate)
        cpn = round(rng.uniform(0.01, 0.05) * 800) / 800
        dirty = dirty_from_curve(bf, cpn, true_df) + rng.normal(0.0, 0.02)  # ~2 cents of noise
        if m % 2:
            quotes.append(BondQuote(bf, cpn, QuoteType.CLEAN, dirty - accrued_interest(bf, cpn)))
        else:
            quotes.append(BondQuote(bf, cpn, QuoteType.YIELD, yield_from_dirty(bf, cpn, dirty)))

    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30.5]
    node_dates = [valdate + dt.timedelta(days=round(365 * x)) for x in tenors]
    for var, itype, method in [('zerorate', 'cubicspline', 'trf'), ('zerorate', 'cubicspline', 'lm'),
                               ('logdiscount', 'cubicspline-linear', 'trf'), ('zerorate', 'pchip', 'trf'),
                               ('zerorate', 'cubicspline', 'SLSQP')]:
        t0 = dt.datetime.now()
        fit = fit_bond_curve(quotes, valdate, node_dates, var, itype, solver=method)
        total = (dt.datetime.now() - t0).total_seconds()
        ye = fit.yield_errors()
        t = np.linspace(0.1, 30, 300)
        curve_err = np.max(np.abs(-np.log(fit.curve.discount_float(t)) / t + np.log(true_df(t)) / t))
        print(f"{var:>11} {itype:<19} {method:<6} total {1e3 * total:6.1f} ms (solve {1e3 * fit.result.elapsed:5.1f}, "
              f"nfev {fit.result.nfev:2d})  rms resid {np.sqrt(np.mean(fit.residuals ** 2)):.3f} bp, "
              f"rms yield err {np.sqrt(np.mean(ye ** 2)):.3f} bp, max zero err {1e4 * curve_err:.2f} bp")
