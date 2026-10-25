import dataclasses
import datetime as dt
import numpy as np
import pytest
from sdevpy.utilities import schedules as sch
from sdevpy.utilities import timegrids
from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.conventions.bondconventions import UST
from sdevpy.rates import bonds, swaps
from sdevpy.rates.bondquotes import settlement_date, bill_flows


VALDATE = dt.datetime(2025, 12, 15)
SETTLE = settlement_date(VALDATE, UST)
MATURITY = dt.date(2036, 2, 15)
CPN = 0.04
BF = bonds.bond_flows(sch.make_bond_scheduler(dt.date(2025, 1, 10), MATURITY, UST), SETTLE, VALDATE)
DIRTY = 98.5 + bonds.accrued_interest(BF, CPN)
SOFR = sch.LegConvention('1Y', DayCount.ACT360, 'USD', pay_lag=2)
# Floating leg rolled like the bond (same calendar and BDC), no payment lag: ends on the bond's last payment date
SOFR_NO_LAG = sch.LegConvention('1Y', DayCount.ACT360, 'USGOV', bdc=BDC.F)


def sofr_curve(t):
    return np.exp(-(0.036 + 0.006 * (1 - np.exp(-t / 5))) * t)


def asw(dirty=DIRTY, conv=SOFR, **kwargs):
    return bonds.asset_swap_spread(BF, CPN, dirty, sofr_curve, MATURITY, conv, VALDATE, **kwargs)


Z = bonds.z_spread(BF, CPN, DIRTY, sofr_curve)


def test_zero_on_the_swap_curve():
    """ A bond priced on the swap curve has no spread to it """
    assert asw(bonds.dirty_from_curve(BF, CPN, sofr_curve), SOFR_NO_LAG) == pytest.approx(0.0, abs=1e-14)


def test_textbook_formula():
    """ Single curve, no payment lag: s = (P_curve - P_mkt) / (100 A) """
    leg = sch.make_leg_schedule(SETTLE, MATURITY, SOFR_NO_LAG)
    d_settle = sofr_curve(np.array([BF.settle_time]))[0]
    a = np.sum(leg.accrual_fraction * sofr_curve(timegrids.model_time(VALDATE, leg.payment))) / d_settle
    p_curve = bonds.dirty_from_curve(BF, CPN, sofr_curve)
    assert asw(conv=SOFR_NO_LAG) == pytest.approx((p_curve - DIRTY) / (100 * a), abs=1e-14)


def test_package_swap_worth_zero():
    """ Independent valuation with swaps.py: upfront (100 - P) + bond coupons - (float + s) on 100 = 0 """
    s = asw()
    leg = sch.make_leg_schedule(SETTLE, MATURITY, SOFR)
    st = swaps.SwapTimes(BF.pay_times, BF.alpha, timegrids.model_time(VALDATE, leg.adj_start),
                         timegrids.model_time(VALDATE, leg.adj_end), timegrids.model_time(VALDATE, leg.payment))
    d_settle = sofr_curve(np.array([BF.settle_time]))[0]
    spread_annuity = np.sum(leg.accrual_fraction * sofr_curve(st.float_pay))
    value = ((100 - DIRTY) * d_settle + 100 * CPN * swaps.annuity(st, sofr_curve)
             - 100 * swaps.float_leg_pv(st, sofr_curve) - 100 * s * spread_annuity)
    assert value == pytest.approx(0.0, abs=1e-10)


def test_close_to_z_spread():
    assert abs(asw() - bonds.z_spread(BF, CPN, DIRTY, sofr_curve)) < 2e-4


def test_proceeds():
    assert asw(proceeds=True) == pytest.approx(asw() * 100 / DIRTY, rel=1e-14)


def test_projection_curve():
    """ Same projection curve = single curve; higher forwards = more floating income = lower spread """
    assert asw(df_proj=sofr_curve) == pytest.approx(asw(), abs=1e-15)
    higher = asw(df_proj=lambda t: sofr_curve(t) * np.exp(-0.001 * t))
    # 10bp cc on ACT/365 = ~10.1-10.3bp of simple ACT/360 floating rate: the spread falls by about that
    assert -0.00105 < higher - asw() < -0.00100


def test_requires_curve_times():
    no_times = dataclasses.replace(BF, pay_times=None)
    with pytest.raises(ValueError):
        bonds.asset_swap_spread(no_times, CPN, DIRTY, sofr_curve, MATURITY, SOFR, VALDATE)


############## SPREAD DURATION ########
def test_spread_duration_matches_finite_differences():
    h = 1e-6
    p = lambda z: bonds.dirty_from_zspread(BF, CPN, sofr_curve, z)
    fd = -(p(Z + h) - p(Z - h)) / (2 * h) / p(Z)
    assert bonds.spread_duration(BF, CPN, sofr_curve, Z) == pytest.approx(fd, rel=1e-8)


def test_spread_duration_of_a_zero_is_its_time():
    """ Single cash flow: spread duration = curve time from settlement to payment """
    bill = bill_flows(dt.date(2026, 6, 16), SETTLE, VALDATE)
    expected = bill.pay_times[0] - bill.settle_time
    assert bonds.spread_duration(bill, 0.0, sofr_curve, 0.01) == pytest.approx(expected, abs=1e-14)


def test_spread_dv01():
    """ 1bp fall in the spread ~ spread DV01 (up to convexity, ~1e-5 relative here) """
    dv01 = bonds.spread_dv01(BF, CPN, sofr_curve, Z)
    assert dv01 == pytest.approx(bonds.spread_duration(BF, CPN, sofr_curve, Z) * DIRTY * 1e-4, rel=1e-10)
    move = bonds.dirty_from_zspread(BF, CPN, sofr_curve, Z - 1e-4) - DIRTY
    assert dv01 == pytest.approx(move, rel=1e-3)


def test_asset_swap_dv01_is_exact():
    """ ASW is linear in the price: a price drop of one ASW DV01 widens the spread by exactly 1bp """
    dv01 = bonds.asset_swap_dv01(BF, sofr_curve, MATURITY, SOFR, VALDATE)
    s0 = bonds.asset_swap_spread(BF, CPN, DIRTY, sofr_curve, MATURITY, SOFR, VALDATE)
    s1 = bonds.asset_swap_spread(BF, CPN, DIRTY - dv01, sofr_curve, MATURITY, SOFR, VALDATE)
    assert s1 - s0 == pytest.approx(1e-4, abs=1e-15)


def test_spread_risk_requires_curve_times():
    no_times = dataclasses.replace(BF, pay_times=None)
    with pytest.raises(ValueError):
        bonds.spread_duration(no_times, CPN, sofr_curve, Z)
    with pytest.raises(ValueError):
        bonds.asset_swap_dv01(no_times, sofr_curve, MATURITY, SOFR, VALDATE)
