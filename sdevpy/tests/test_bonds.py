import datetime as dt
import numpy as np
import pytest
from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.utilities import schedules as sch
from sdevpy.utilities import timegrids
from sdevpy.rates import bonds
from sdevpy.utilities.schedules import make_bond_scheduler
from sdevpy.rates.bonds import bond_flows, dirty_from_curve, flatten, dirty_prices
from sdevpy.rates.bondquotes import bill_flows
from sdevpy.calibration.rates import yieldcurve as ycrv


VALDATE = dt.datetime(2026, 10, 26)
UST = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'USD', bdc=BDC.F, adjust_accrual=False, eom=True)
GILT = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'GBP', bdc=BDC.F, adjust_accrual=False)
BOND = sch.make_bond_scheduler(dt.date(2026, 2, 15), dt.date(2036, 2, 15), UST)
COUPON = 0.04


def test_flows_on_coupon_date():
    # Coupon paid on the settlement date goes to the seller: nothing accrued, next coupon a full period away
    bf = bonds.bond_flows(BOND, dt.date(2026, 8, 15))
    assert len(bf.payment) == 19
    assert bf.accrued_fraction == 0.0
    assert np.allclose(bf.periods, np.arange(1, 20), rtol=0, atol=1e-12)
    assert abs(bonds.dirty_from_yield(bf, COUPON, COUPON) - 100.0) < 1e-10  # yield = coupon -> par


def test_accrued_interest():
    bf = bonds.bond_flows(BOND, dt.date(2026, 5, 15))  # 89 of 181 days since 15-Feb
    assert abs(bonds.accrued_interest(bf, COUPON) - 100 * COUPON * 0.5 * 89 / 181) < 1e-12
    assert abs(bf.periods[0] - 92 / 181) < 1e-12


def test_cashflows():
    bf = bonds.bond_flows(BOND, dt.date(2026, 5, 15))
    cf = bonds.cashflows(bf, COUPON)
    assert np.allclose(cf[:-1], 2.0, rtol=0, atol=1e-12)
    assert abs(cf[-1] - 102.0) < 1e-12
    assert bf.payment[0] == np.datetime64('2026-08-17')  # 15-Aug-2026 is a Saturday


def test_dirty_from_yield_treasury_formula():
    # US Treasury formula: P = sum_k CF_k / (1 + y/2)^(k + r/s), r = days to next coupon, s = days in period
    settle, y = dt.date(2026, 5, 15), 0.045
    bf = bonds.bond_flows(BOND, settle)
    r, s = 92, 181
    n = len(bf.payment)
    ref = sum((2.0 if k < n - 1 else 102.0) / (1 + y / 2) ** (k + r / s) for k in range(n))
    assert abs(bonds.dirty_from_yield(bf, COUPON, y) - ref) < 1e-10


def test_yield_round_trip():
    bf = bonds.bond_flows(BOND, dt.date(2026, 5, 15))
    for y in (0.0, 0.01, 0.045, 0.12):
        dirty = bonds.dirty_from_yield(bf, COUPON, y)
        assert abs(bonds.yield_from_dirty(bf, COUPON, dirty) - y) < 1e-10


def test_stub_accrued():
    # Issued 10-Jan-2026, short first coupon 15-Feb-2026, regular half-year 15-Aug-25 -> 15-Feb-26 (184 days)
    stub_bond = sch.make_bond_scheduler(dt.date(2026, 1, 10), dt.date(2031, 2, 15), GILT)
    bf = bonds.bond_flows(stub_bond, dt.date(2026, 1, 20))
    assert abs(bf.accrued_fraction - 10 / 368) < 1e-12
    assert abs(bf.periods[0] - 26 / 184) < 1e-12


def test_matured_bond_raises():
    with pytest.raises(ValueError):
        bonds.bond_flows(BOND, dt.date(2036, 3, 1))


def test_curve_pricing_requires_valdate():
    bf = bonds.bond_flows(BOND, dt.date(2026, 5, 15))
    with pytest.raises(ValueError):
        bonds.dirty_from_curve(bf, COUPON, lambda t: np.ones_like(t))


def test_curve_price_and_par_coupon():
    from sdevpy.calibration.rates import yieldcurve as ycrv
    valdate = dt.datetime(2026, 5, 14)
    zdates = [dt.datetime(2027, 5, 14), dt.datetime(2031, 5, 14), dt.datetime(2036, 5, 14)]
    zrs = np.array([0.035, 0.037, 0.040])
    curve = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var='zerorate', interp_type='linear')
    curve.set_data(zdates, np.exp(-zrs * timegrids.model_time(valdate, zdates)))
    df = curve.discount_float

    bf = bonds.bond_flows(BOND, dt.date(2026, 5, 15), valdate)
    dirty = bonds.dirty_from_curve(bf, COUPON, df)
    assert abs(dirty - 101.048973209) < 1e-8  # regression
    # Discounting forward to settlement: same as PV at valdate divided by P(settlement)
    pv = np.sum(bonds.cashflows(bf, COUPON) * df(bf.pay_times))
    assert abs(dirty - pv / df(np.array([1 / 365]))[0]) < 1e-12

    par = bonds.par_coupon(bf, df)
    clean = bonds.dirty_from_curve(bf, par, df) - bonds.accrued_interest(bf, par)
    assert abs(clean - 100.0) < 1e-10


################ FOR CURVE CONSTRUCTION ###############################################################################

def make_curve():
    dates = [dt.datetime(2027, 10, 26), dt.datetime(2031, 10, 26), dt.datetime(2036, 10, 26)]
    zrs = np.array([0.035, 0.037, 0.040])
    curve = ycrv.InterpolatedYieldCurve(valdate=VALDATE, interp_var='zerorate', interp_type='cubicspline')
    curve.set_data(dates, np.exp(-zrs * timegrids.model_time(VALDATE, dates)))
    return curve


def make_instruments():
    """ Bill, note and bond, with different settlement dates """
    s1, s2 = dt.date(2026, 10, 27), dt.date(2026, 10, 28)
    flows = [bill_flows(dt.date(2027, 1, 26), s1, VALDATE),
             bond_flows(make_bond_scheduler(dt.date(2026, 8, 15), dt.date(2036, 8, 15), UST), s2, VALDATE),
             bond_flows(make_bond_scheduler(dt.date(2025, 2, 28), dt.date(2030, 2, 28), UST), s1, VALDATE)]
    return flows, [0.0, 0.04125, 0.0375]


def test_flatten_layout():
    flows, coupons = make_instruments()
    ff = flatten(flows, coupons)
    sizes = [len(bf.alpha) for bf in flows]
    assert ff.n_bonds == 3 and ff.n_flows == sum(sizes)
    np.testing.assert_array_equal(ff.starts, [0, sizes[0], sizes[0] + sizes[1]])
    np.testing.assert_array_equal(np.bincount(ff.owner), sizes)
    assert len(ff.times) == ff.n_flows + ff.n_bonds
    assert ff.amounts[0] == 100.0  # bill: redemption only
    assert ff.amounts[ff.starts[2] - 1] == pytest.approx(100.0 + 100.0 * 0.04125 * flows[1].alpha[-1])


def test_flat_prices_match_per_bond():
    flows, coupons = make_instruments()
    curve = make_curve()
    flat = dirty_prices(flatten(flows, coupons), curve.discount_float)
    loop = [dirty_from_curve(bf, c, curve.discount_float) for bf, c in zip(flows, coupons, strict=True)]
    np.testing.assert_allclose(flat, loop, rtol=0, atol=1e-12)


def test_redemptions():
    flows, coupons = make_instruments()
    curve = make_curve()
    flat = dirty_prices(flatten(flows, coupons, [100.0, 101.0, 100.0]), curve.discount_float)
    assert flat[1] == pytest.approx(dirty_from_curve(flows[1], coupons[1], curve.discount_float, 101.0), abs=1e-12)


def test_flatten_needs_valdate():
    bf = bill_flows(dt.date(2027, 1, 26), dt.date(2026, 10, 27))
    with pytest.raises(ValueError):
        flatten([bf], [0.0])
