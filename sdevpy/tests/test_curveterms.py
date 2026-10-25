import datetime as dt
import numpy as np
import pytest
from sdevpy.utilities import timegrids
from sdevpy.utilities.schedules import make_bond_scheduler
from sdevpy.rates.bonds import bond_flows, dirty_from_curve, accrued_interest
from sdevpy.rates.bondquotes import QuoteType, BondQuote, bill_flows, discount_rate_from_price
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
from sdevpy.maths.leastsquares import create_lsq_solver
from sdevpy.calibration.rates.curveterms import ForwardGrid, SmoothnessPenalty, ForwardFloor
from sdevpy.calibration.bonds.bondcurve import fit_bond_curve, default_nodes
from sdevpy.conventions.bondconventions import UST


VALDATE = dt.datetime(2026, 10, 26)
SETTLE = dt.date(2026, 10, 27)
# UST = LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.5]
NODE_DATES = [VALDATE + dt.timedelta(days=round(365 * x)) for x in TENORS]
NODE_TIMES = timegrids.model_time(VALDATE, NODE_DATES)
DENSE_DATES = [VALDATE + dt.timedelta(days=round(365 * x)) for x in [0.25, 0.5] + list(range(1, 31))]
ITYPE = 'cubicspline-linear'


def true_df(t):
    x = np.maximum(t, 1e-8) / 2.0
    f = (1.0 - np.exp(-x)) / x
    return np.exp(-(0.048 - 0.010 * f - 0.020 * (f - np.exp(-x))) * t)


def make_quotes(noise=0.0, max_years=30, seed=1):
    rng = np.random.default_rng(seed)
    quotes = []
    for k in range(8):
        bf = bill_flows(SETTLE + dt.timedelta(days=28 * (k + 1)), SETTLE, VALDATE)
        quotes.append(BondQuote(bf, 0.0, QuoteType.DISCOUNT, discount_rate_from_price(bf, dirty_from_curve(bf, 0.0, true_df))))
    for m in range(0, 12 * max_years, 3):
        mat = dt.date(2027 + m // 12, 1 + m % 12, 15)
        bf = bond_flows(make_bond_scheduler(dt.date(mat.year - 30, mat.month, 15), mat, UST), SETTLE, VALDATE)
        cpn = 0.0125 + 0.0025 * (m % 13)
        dirty = dirty_from_curve(bf, cpn, true_df) + noise * rng.normal()
        quotes.append(BondQuote(bf, cpn, QuoteType.CLEAN, dirty - accrued_interest(bf, cpn)))
    return quotes


def grid_forwards(curve, horizon, step=1 / 12):
    t = np.linspace(0.0, horizon, int(np.ceil(horizon / step)) + 1)
    return -np.diff(np.log(curve.discount_float(t))) / (t[1] - t[0])


################## Forward grid and penalty #######################################################
@pytest.mark.parametrize("var, itype", [('zerorate', 'cubicspline'), ('logdiscount', 'linear-linear'),
                                        ('discount', 'cubicspline-linear')])
def test_forward_grid(var, itype):
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var=var, interp_type=itype)
    curve.set_data(NODE_DATES, true_df(NODE_TIMES))
    y = curve.node_values(NODE_TIMES, true_df(NODE_TIMES))
    expected = grid_forwards(curve, NODE_TIMES[-1])  # before ForwardGrid: building its basis overwrites the curve
    fg = ForwardGrid(curve, NODE_TIMES, NODE_TIMES[-1], 1 / 12)
    f, jac = fg.forwards_jac(y)
    np.testing.assert_allclose(f, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fg.forwards(y), f, rtol=0, atol=1e-15)
    h = 1e-7
    fd = np.column_stack([(fg.forwards(y + h * e) - fg.forwards(y - h * e)) / (2 * h) for e in np.eye(len(y))])
    assert np.max(np.abs(jac - fd)) < 1e-6 * np.max(np.abs(fd))


@pytest.mark.parametrize("order", [1, 2])
def test_penalty_rows(order):
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var='zerorate', interp_type=ITYPE)
    y = curve.node_values(NODE_TIMES, true_df(NODE_TIMES))
    rows = SmoothnessPenalty(0.01, order=order).build(curve, NODE_TIMES)
    r, jac = rows(y)
    h = 1e-7
    fd = np.column_stack([(rows(y + h * e)[0] - rows(y - h * e)[0]) / (2 * h) for e in np.eye(len(y))])
    assert np.max(np.abs(jac - fd)) < 1e-6 * np.max(np.abs(fd))
    # Constant lam = constant function; flat curve has no roughness
    r2, _ = SmoothnessPenalty(lambda t: np.full(len(t), 0.01), order=order).build(curve, NODE_TIMES)(y)
    np.testing.assert_array_equal(r, r2)
    r_flat, _ = rows(curve.node_values(NODE_TIMES, np.exp(-0.04 * NODE_TIMES)))
    assert np.max(np.abs(r_flat)) < 1e-6
    with pytest.raises(ValueError):
        SmoothnessPenalty(0.01, order=3).build(curve, NODE_TIMES)


################## Fits with penalty / floor ######################################################
def test_tiny_penalty_changes_nothing():
    quotes = make_quotes(noise=0.03)
    ref = fit_bond_curve(quotes, VALDATE, NODE_DATES, interp_type=ITYPE)
    fit = fit_bond_curve(quotes, VALDATE, NODE_DATES, interp_type=ITYPE, penalty=SmoothnessPenalty(1e-12))
    np.testing.assert_allclose(fit.curve.dfs, ref.curve.dfs, rtol=0, atol=1e-9)
    assert len(fit.residuals) == len(quotes) < len(fit.result.residuals)


@pytest.mark.parametrize("order, tol", [(1, 2e-4), (2, 5e-5)])
def test_strong_penalty_shapes_forwards(order, tol):
    """ Huge lam: forwards close to flat (order 1) or straight (order 2) """
    fit = fit_bond_curve(make_quotes(noise=0.03), VALDATE, DENSE_DATES, interp_type=ITYPE,
                         penalty=SmoothnessPenalty(1e4, order=order))
    f = grid_forwards(fit.curve, fit.node_times[-1])
    assert np.max(np.abs(np.diff(f, n=order))) < tol


def test_penalty_reduces_noise_with_dense_nodes():
    """ 32 nodes on noisy quotes: the penalty brings the forwards closer to the truth """
    quotes = make_quotes(noise=0.03)
    t_mid = np.linspace(0, 30, 361)[:-1] + 1 / 24
    true_f = -np.diff(np.log(true_df(np.linspace(0, 30, 361)))) * 12
    errs = []
    for pen in [None, SmoothnessPenalty(1e-2)]:
        fit = fit_bond_curve(quotes, VALDATE, DENSE_DATES, interp_type=ITYPE, penalty=pen)
        f = -np.diff(np.log(fit.curve.discount_float(np.linspace(0, 30, 361)))) * 12
        errs.append(np.max(np.abs(f - true_f)[t_mid > 0.5]))
    assert errs[1] < 0.5 * errs[0]


def test_penalty_determines_nodes_without_quotes():
    """ Quotes to 10Y only, nodes to 30Y: singular without penalty, determined with it (flat forwards beyond) """
    quotes = make_quotes(max_years=10)
    fit = fit_bond_curve(quotes, VALDATE, DENSE_DATES, interp_type=ITYPE, penalty=SmoothnessPenalty(1e-2, order=1))
    assert fit.result.success
    assert np.max(np.abs(fit.residuals)) < 1.0  # spline vs Nelson-Siegel truth, not the penalty
    f = grid_forwards(fit.curve, fit.node_times[-1])
    assert np.ptp(f[-120:]) < 1e-4  # last 10Y nearly flat


def test_forward_floor():
    quotes = make_quotes()
    solver = create_lsq_solver('SLSQP', ftol=1e-14)
    free = fit_bond_curve(quotes, VALDATE, NODE_DATES, interp_type=ITYPE, solver=solver)
    f_free = grid_forwards(free.curve, free.node_times[-1])
    # Non-binding floor: same fit
    loose = fit_bond_curve(quotes, VALDATE, NODE_DATES, interp_type=ITYPE, solver=solver,
                           constraints=[ForwardFloor(f_free.min() - 0.005)])
    np.testing.assert_allclose(loose.curve.dfs, free.curve.dfs, rtol=0, atol=1e-8)
    # Binding floor
    floor = 0.040
    assert f_free.min() < floor
    fit = fit_bond_curve(quotes, VALDATE, NODE_DATES, interp_type=ITYPE, solver=solver,
                         constraints=[ForwardFloor(floor)])
    f = grid_forwards(fit.curve, fit.node_times[-1])
    assert f.min() >= floor - 1e-9
    assert f.min() == pytest.approx(floor, abs=1e-8)


################## Nodes ##########################################################################
def test_default_nodes():
    quotes = make_quotes()
    nodes = default_nodes(VALDATE, quotes)
    mats = sorted(q.flows.payment[-1] for q in quotes)
    node_d = [np.datetime64(n.date(), 'D') for n in nodes]
    assert node_d[-1] == mats[-1]
    assert all(a < b for a, b in zip(node_d, node_d[1:]))
    prev = np.datetime64(VALDATE.date(), 'D')
    for n in node_d:  # every interval has a maturity
        assert any(prev < m <= n for m in mats)
        prev = n


def test_default_nodes_merges_empty_intervals():
    """ Maturities only around 1Y and 8.5-9.5Y: 2Y-7Y nodes dropped, nodes at 1Y and the last maturity """
    def years(q):
        return (q.flows.payment[-1] - np.datetime64(SETTLE)).astype(int) / 365
    quotes = [q for q in make_quotes() if 0.9 < years(q) < 1.1 or 8.5 < years(q) < 9.5]
    nodes = default_nodes(VALDATE, quotes)
    assert len(nodes) == 2
    assert (nodes[0] - VALDATE).days == 365
    assert np.datetime64(nodes[1].date(), 'D') == max(q.flows.payment[-1] for q in quotes)
