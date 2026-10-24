import datetime as dt
import numpy as np
import pytest
from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.utilities import schedules as sch
from sdevpy.utilities import timegrids
from sdevpy.rates import bonds


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
