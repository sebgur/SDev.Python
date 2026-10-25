import datetime as dt
import numpy as np
import pytest
from sdevpy.utilities import timegrids
from sdevpy.utilities.schedules import make_bond_scheduler
from sdevpy.rates.bonds import bond_flows, flatten, dirty_prices, prices_from_discounts, prices_jac
from sdevpy.rates.bondquotes import bill_flows
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
from sdevpy.calibration.rates.curvebasis import CurveBasis
from sdevpy.conventions.bondconventions import UST


VALDATE = dt.datetime(2026, 10, 26)
# UST = LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30.5]
NODE_DATES = [VALDATE + dt.timedelta(days=round(365 * x)) for x in TENORS]
NODE_TIMES = timegrids.model_time(VALDATE, NODE_DATES)
NODE_DFS = np.exp(-(0.035 + 0.01 * (1 - np.exp(-NODE_TIMES / 5))) * NODE_TIMES)
LINEAR = [('zerorate', 'linear'), ('zerorate', 'cubicspline'), ('logdiscount', 'cubicspline-linear'),
          ('discount', 'bspline-linear'), ('logdiscount', 'linear-linear')]
NONLINEAR = [('zerorate', 'pchip'), ('logdiscount', 'akima-linear')]


def make_flat_flows():
    s1, s2 = dt.date(2026, 10, 27), dt.date(2026, 10, 28)
    flows = [bill_flows(dt.date(2027, 1, 26), s1, VALDATE),
             bill_flows(dt.date(2027, 4, 27), s2, VALDATE),
             bond_flows(make_bond_scheduler(dt.date(2025, 2, 28), dt.date(2030, 2, 28), UST), s1, VALDATE),
             bond_flows(make_bond_scheduler(dt.date(2026, 8, 15), dt.date(2036, 8, 15), UST), s2, VALDATE),
             bond_flows(make_bond_scheduler(dt.date(2026, 8, 15), dt.date(2056, 8, 15), UST), s1, VALDATE)]
    return flatten(flows, [0.0, 0.0, 0.0375, 0.04125, 0.0475])


@pytest.mark.parametrize("var, itype", LINEAR + NONLINEAR)
def test_set_nodes_roundtrip(var, itype):
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var=var, interp_type=itype)
    curve.set_nodes(NODE_TIMES, curve.node_values(NODE_TIMES, NODE_DFS))
    np.testing.assert_allclose(curve.discount_float(NODE_TIMES), NODE_DFS, rtol=0, atol=1e-14)


@pytest.mark.parametrize("var, itype, linear", [(v, i, True) for v, i in LINEAR] + [(v, i, False) for v, i in NONLINEAR])
def test_linearity_detection(var, itype, linear):
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var=var, interp_type=itype)
    assert CurveBasis(curve, NODE_TIMES, make_flat_flows().times).linear == linear


@pytest.mark.parametrize("var, itype", LINEAR + NONLINEAR)
def test_basis_prices_match_curve(var, itype):
    """ Prices from the basis = prices on the final curve built with set_data """
    ff = make_flat_flows()
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var=var, interp_type=itype)
    cb = CurveBasis(curve, NODE_TIMES, ff.times)
    p = prices_from_discounts(ff, cb.discount(curve.node_values(NODE_TIMES, NODE_DFS)))
    curve.set_data(NODE_DATES, NODE_DFS)
    np.testing.assert_allclose(p, dirty_prices(ff, curve.discount_float), rtol=0, atol=1e-12)


@pytest.mark.parametrize("var, itype, h, tol", [(v, i, 1e-6, 1e-8) for v, i in LINEAR]
                                              + [(v, i, 1e-9, 1e-4) for v, i in NONLINEAR])
def test_jacobian_vs_finite_differences(var, itype, h, tol):
    ff = make_flat_flows()
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var=var, interp_type=itype)
    cb = CurveBasis(curve, NODE_TIMES, ff.times)
    y = curve.node_values(NODE_TIMES, NODE_DFS)
    p, jac = prices_jac(ff, *cb.discount_jac(y))
    np.testing.assert_allclose(p, prices_from_discounts(ff, cb.discount(y)), rtol=0, atol=1e-13)
    fd = np.column_stack([(prices_from_discounts(ff, cb.discount(y + h * e))
                           - prices_from_discounts(ff, cb.discount(y - h * e))) / (2 * h) for e in np.eye(len(y))])
    assert np.max(np.abs(jac - fd)) < tol * np.max(np.abs(fd))
