import datetime as dt
import numpy as np
import pytest
import scipy.optimize as opt
from sdevpy.utilities import timegrids
from sdevpy.utilities.schedules import make_bond_scheduler
from sdevpy.rates.bonds import bond_flows, dirty_from_curve, accrued_interest, yield_from_dirty
from sdevpy.rates.bondquotes import QuoteType, BondQuote, bill_flows, discount_rate_from_price
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
from sdevpy.maths.leastsquares import create_lsq_solver
from sdevpy.rates.bonds import dirty_prices
from sdevpy.calibration.bonds.bondcurve import fit_bond_curve, default_nodes
from sdevpy.rates.syntheticbonds import sample_bond_universe
from sdevpy.conventions.bondconventions import UST


VALDATE = dt.datetime(2026, 10, 26)
SETTLE = dt.date(2026, 10, 27)
# UST = LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30.5]
NODE_DATES = [VALDATE + dt.timedelta(days=round(365 * x)) for x in TENORS]
NODE_TIMES = timegrids.model_time(VALDATE, NODE_DATES)
PER_END = ('not-a-knot', 'natural')


def true_curve(interp_var='zerorate', interp_type='cubicspline'):
    """ Curve in the same family as the fit: exactly recoverable from enough quotes """
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var=interp_var, interp_type=interp_type)
    zr = 0.038 - 0.004 * np.exp(-NODE_TIMES / 2) + 0.006 * (1 - np.exp(-NODE_TIMES / 8))
    curve.set_data(NODE_DATES, np.exp(-zr * NODE_TIMES))
    return curve


def make_quotes(df, noise=0.0, seed=1):
    """ Bills on discount rate; notes/bonds alternately on clean price and yield; settlement T+1 or T+2 """
    rng = np.random.default_rng(seed)
    quotes = []
    for k in range(8):
        settle = SETTLE + dt.timedelta(days=k % 2)
        bf = bill_flows(SETTLE + dt.timedelta(days=28 * (k + 1)), settle, VALDATE)
        quotes.append(BondQuote(bf, 0.0, QuoteType.DISCOUNT, discount_rate_from_price(bf, dirty_from_curve(bf, 0.0, df))))
    for m in range(0, 360, 6):
        mat = dt.date(2027 + m // 12, 2 + m % 12, 15)
        settle = SETTLE + dt.timedelta(days=m % 2)
        bf = bond_flows(make_bond_scheduler(dt.date(mat.year - 30, mat.month, 15), mat, UST), settle, VALDATE)
        cpn = 0.0125 + 0.0025 * (m % 13)
        dirty = dirty_from_curve(bf, cpn, df) + noise * rng.normal()
        if m % 12:
            quotes.append(BondQuote(bf, cpn, QuoteType.CLEAN, dirty - accrued_interest(bf, cpn)))
        else:
            quotes.append(BondQuote(bf, cpn, QuoteType.YIELD, yield_from_dirty(bf, cpn, dirty)))
    return quotes


################## Fitter #########################################################################
@pytest.mark.parametrize("var, itype", [('zerorate', 'cubicspline'), ('zerorate', 'linear'),
                                        ('logdiscount', 'cubicspline-linear'), ('discount', 'bspline-linear'),
                                        ('zerorate', 'pchip')])
def test_exact_recovery(var, itype):
    """ Quotes generated on a curve of the fitted family: the fit recovers its node discount factors """
    truth = true_curve(var, itype)
    fit = fit_bond_curve(make_quotes(truth.discount_float), VALDATE, NODE_DATES, var, itype, bc_type=None)
    assert fit.result.success
    np.testing.assert_allclose(fit.curve.dfs, truth.dfs, rtol=0, atol=1e-10)
    assert np.max(np.abs(fit.residuals)) < 1e-6
    assert fit.curve.dates == NODE_DATES


def test_model_prices_on_final_curve():
    fit = fit_bond_curve(make_quotes(true_curve().discount_float, noise=0.03), VALDATE, NODE_DATES)
    np.testing.assert_allclose(dirty_prices(fit.flat_flows, fit.curve.discount_float), fit.model_prices,
                               rtol=0, atol=1e-10)


def test_dv01_weights_give_bp_residuals():
    """ Weighted price residuals ~ exact yield errors in bp """
    fit = fit_bond_curve(make_quotes(true_curve().discount_float, noise=0.03), VALDATE, NODE_DATES)
    ye = fit.yield_errors()
    assert np.sqrt(np.mean(ye ** 2)) > 0.1  # noise is visible
    np.testing.assert_allclose(fit.residuals, ye, rtol=0, atol=1e-3)


@pytest.mark.parametrize("solver", ['trf', 'lm', 'dogbox', create_lsq_solver('SLSQP', ftol=1e-14),
                                    create_lsq_solver('trf', xtol=1e-14)])
def test_solvers_agree(solver):
    quotes = make_quotes(true_curve().discount_float, noise=0.03)
    ref = fit_bond_curve(quotes, VALDATE, NODE_DATES)
    fit = fit_bond_curve(quotes, VALDATE, NODE_DATES, solver=solver)
    np.testing.assert_allclose(fit.curve.dfs, ref.curve.dfs, rtol=0, atol=1e-9)


def test_weights_options():
    quotes = make_quotes(true_curve().discount_float, noise=0.03)
    unit = fit_bond_curve(quotes, VALDATE, NODE_DATES, weights=None)
    np.testing.assert_array_equal(unit.weights, 1.0)
    np.testing.assert_allclose(unit.residuals, unit.target_prices - unit.model_prices)
    given = fit_bond_curve(quotes, VALDATE, NODE_DATES, weights=np.full(len(quotes), 2.0))
    np.testing.assert_allclose(given.curve.dfs, unit.curve.dfs, rtol=0, atol=1e-10)  # scale-invariant
    with pytest.raises(ValueError):
        fit_bond_curve(quotes, VALDATE, NODE_DATES, weights='bogus')


def test_bounds():
    """ Cap all zero rates at 4%: the binding nodes sit on the bound """
    quotes = make_quotes(true_curve().discount_float)
    n = len(NODE_DATES)
    fit = fit_bond_curve(quotes, VALDATE, NODE_DATES, interp_var='zerorate',
                         bounds=opt.Bounds(np.zeros(n), np.full(n, 0.04)))
    zr = -np.log(fit.curve.dfs) / NODE_TIMES
    assert np.all(zr <= 0.04 + 1e-12)
    assert np.max(zr) == pytest.approx(0.04, abs=1e-10)


def test_invalid_inputs():
    quotes = make_quotes(true_curve().discount_float)
    with pytest.raises(ValueError):
        fit_bond_curve(quotes, VALDATE, NODE_DATES[::-1])
    no_valdate = [BondQuote(bill_flows(dt.date(2027, 1, 26), SETTLE), 0.0, QuoteType.DISCOUNT, 0.04)]
    with pytest.raises(ValueError):
        fit_bond_curve(no_valdate, VALDATE, NODE_DATES)


def test_fit_default_long_end():
    """ Default fit (log-DF, per-end spline): forwards within 1bp of the truth, flat at the last node """
    u = sample_bond_universe(VALDATE, seed=42)
    nodes = default_nodes(VALDATE, u['quotes'])
    fit = fit_bond_curve(u['quotes'], VALDATE, nodes)
    assert fit.curve.bc_type == PER_END
    h, T = 1 / 12, fit.node_times[-1]
    t = np.linspace(h, T - h / 2, 300)
    fwd = -np.log(fit.curve.discount_float(t + h / 2) / fit.curve.discount_float(t - h / 2)) / h
    true = -np.log(u['target_curve'](t + h / 2) / u['target_curve'](t - h / 2)) / h
    assert 1e4 * np.max(np.abs(fwd - true)) < 1.5
    assert 1e4 * np.max(np.abs(fwd - true)[t > 20]) < 1.0


@pytest.mark.parametrize("interp_var, interp_type", [('zerorate', 'cubicspline-linear'), ('logdiscount', 'pchip-linear'),
                                                     ('logdiscount', 'linear-linear')])
def test_fit_auto_bc_elsewhere(interp_var, interp_type):
    u = sample_bond_universe(VALDATE, seed=42)
    fit = fit_bond_curve(u['quotes'], VALDATE, default_nodes(VALDATE, u['quotes']), interp_var, interp_type)
    assert fit.curve.bc_type is None
    assert fit.result.success
