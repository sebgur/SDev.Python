import datetime as dt
import numpy as np
import dataclasses
import pytest
from sdevpy.utilities import schedules as sch
from sdevpy.utilities import timegrids
from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.rates import bonds
from sdevpy.utilities.schedules import make_bond_scheduler, LegConvention, BondConvention
from sdevpy.rates.bonds import (bond_flows, dirty_from_curve, flatten, dirty_prices,
    yields_from_dirty, dirty_from_yields, dv01s, dirty_from_yield, par_coupon, convert_yield,
    par_yield, g_spread)
from sdevpy.rates.bondquotes import bill_flows
from sdevpy.calibration.rates import yieldcurve as ycrv
from sdevpy.conventions.bondconventions import UST, GILT


COMPOUND = dataclasses.replace(UST, simple_final_period=False)
VALDATEOLD = dt.datetime(2025, 12, 15)
SETTLE = dt.date(2025, 12, 16)
CORP = BondConvention(LegConvention('1Y', DayCount.D30_360, 'USD', bdc=BDC.F, adjust_accrual=False))

VALDATE = dt.datetime(2026, 10, 26)
# UST = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
# GILT = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'GBP', bdc=BDC.F, adjust_accrual=False)
BOND = sch.make_bond_scheduler(dt.date(2026, 2, 15), dt.date(2036, 2, 15), UST)
COUPON = 0.04


def final_period_note(conv=UST):
    """ 4.25% 15-Feb-2026 on 16-Dec-2025: only the last coupon left, 61 of the 184 days of the period to run """
    return bonds.bond_flows(sch.make_bond_scheduler(dt.date(2023, 2, 15), dt.date(2026, 2, 15), conv), SETTLE, VALDATE)


def test_treasury_formula():
    """ Dirty price = (100 + C/2) / (1 + (r/s) y/2), r = days to maturity, s = days in the period """
    bf = final_period_note()
    assert bf.simple_flows.tolist() == [True]
    assert bonds.dirty_from_yield(bf, 0.0425, 0.04) == pytest.approx(102.125 / (1 + 0.02 * 61 / 184), abs=1e-12)
    compound = final_period_note(COMPOUND)
    assert not compound.simple_flows.any()
    assert bonds.dirty_from_yield(compound, 0.0425, 0.04) == pytest.approx(102.125 * 1.02 ** (-61 / 184), abs=1e-12)


def gov_df(t):
    return np.exp(-(0.036 + 0.012 * (1 - np.exp(-t / 4))) * t)


def test_convert_yield():
    assert convert_yield(0.04, 1, 2) == pytest.approx(2 * (np.sqrt(1.04) - 1), abs=1e-15)
    assert convert_yield(convert_yield(0.05, 2, 12), 12, 2) == pytest.approx(0.05, abs=1e-15)
    assert convert_yield(0.05, 2, 2) == 0.05


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
    assert np.all(np.diff(ff.times) > 0)  # distinct, increasing
    np.testing.assert_array_equal(ff.times[ff.flow_index], np.concatenate([bf.pay_times for bf in flows]))
    np.testing.assert_array_equal(ff.times[ff.settle_index], [bf.settle_time for bf in flows])
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


def test_vectorized_yields_and_dv01():
    flows, coupons = make_instruments()
    ff = flatten(flows, coupons)
    dirty = dirty_prices(ff, make_curve().discount_float)
    y = yields_from_dirty(ff, dirty)
    expected = [bonds.yield_from_dirty(bf, c, p) for bf, c, p in zip(flows, coupons, dirty, strict=True)]
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-13)
    np.testing.assert_allclose(dirty_from_yields(ff, y), dirty, rtol=0, atol=1e-10)
    np.testing.assert_allclose(dv01s(ff, y), [bonds.dv01(bf, c, yi) for bf, c, yi in zip(flows, coupons, y, strict=True)],
                               rtol=1e-13)


@pytest.mark.parametrize("maturity", [dt.date(2026, 5, 15), dt.date(2030, 11, 15), dt.date(2036, 2, 15),
                                      dt.date(2055, 8, 15)])
def test_par_yield(maturity):
    """ The hypothetical bond issued at settlement prices at par at that yield, close to its coupon """
    y = par_yield(maturity, SETTLE, UST, gov_df, VALDATEOLD)
    bf = bond_flows(make_bond_scheduler(SETTLE, maturity, UST), SETTLE, VALDATEOLD)
    c = par_coupon(bf, gov_df)
    assert dirty_from_yield(bf, c, y) == pytest.approx(100.0, abs=1e-10)
    assert dirty_from_curve(bf, c, gov_df) == pytest.approx(100.0, abs=1e-10)
    assert abs(y - c) < 1e-4  # street yield != coupon over the short first period


def test_g_spread_of_par_government_bond_is_zero():
    maturity = dt.date(2036, 2, 15)
    bf = bond_flows(make_bond_scheduler(SETTLE, maturity, UST), SETTLE, VALDATEOLD)
    c = par_coupon(bf, gov_df)
    assert g_spread(bf, c, 100.0, gov_df, maturity, UST, VALDATEOLD) == pytest.approx(0.0, abs=1e-12)


def test_g_spread_of_corporate():
    """ Annual 30/360 bond: ~0 when priced on the government curve, ~ the spread when priced above it """
    maturity = dt.date(2035, 11, 15)
    bf = bond_flows(make_bond_scheduler(dt.date(2025, 11, 15), maturity, CORP), SETTLE, VALDATEOLD)
    on_curve = dirty_from_curve(bf, 0.05, gov_df)
    assert abs(g_spread(bf, 0.05, on_curve, gov_df, maturity, UST, VALDATEOLD)) < 1e-4  # coupon effect only
    gs = [g_spread(bf, 0.05, dirty_from_curve(bf, 0.05, lambda t, z=z: gov_df(t) * np.exp(-z * t)), gov_df,
                   maturity, UST, VALDATEOLD) for z in (0.005, 0.01, 0.02)]
    assert gs[0] < gs[1] < gs[2]
    assert gs[1] == pytest.approx(0.01, abs=5e-4)  # cc spread vs semi-annual yield spread


@pytest.mark.parametrize("conv", [UST, COMPOUND])
def test_yield_round_trip(conv):
    bf = final_period_note(conv)
    dirty = bonds.dirty_from_yield(bf, 0.0425, 0.04)
    assert bonds.yield_from_dirty(bf, 0.0425, dirty) == pytest.approx(0.04, abs=1e-13)


def test_more_than_one_coupon_left_is_compound():
    """ Same note one period earlier (two coupons left): the convention changes nothing """
    for conv in (UST, COMPOUND):
        bf = bonds.bond_flows(sch.make_bond_scheduler(dt.date(2023, 2, 15), dt.date(2026, 2, 15), conv),
                              dt.date(2025, 6, 16), dt.datetime(2025, 6, 13))
        assert not bf.simple_flows.any()
        assert bonds.dirty_from_yield(bf, 0.0425, 0.04) == pytest.approx(
            float(np.sum(bonds.cashflows(bf, 0.0425) * 1.02 ** -bf.periods)), abs=1e-12)


@pytest.mark.parametrize("conv", [UST, COMPOUND])
def test_risk_matches_finite_differences(conv):
    bf, c, y, h = final_period_note(conv), 0.0425, 0.04, 1e-5
    p = lambda yy: bonds.dirty_from_yield(bf, c, yy)
    assert bonds.modified_duration(bf, c, y) == pytest.approx(-(p(y + h) - p(y - h)) / (2 * h) / p(y), rel=1e-8)
    h2 = 1e-3  # second difference: larger step, rounding dominates at 1e-5
    assert bonds.convexity(bf, c, y) == pytest.approx((p(y + h2) - 2 * p(y) + p(y - h2)) / h2 ** 2 / p(y), rel=1e-6)
    assert bonds.dv01(bf, c, y) == pytest.approx(bonds.modified_duration(bf, c, y) * p(y) * 1e-4, rel=1e-14)
    assert bonds.macaulay_duration(bf, c, y) == pytest.approx(61 / 184 / 2, abs=1e-14)  # one flow: time to it


def test_vectorized_matches_single_bond():
    """ Mix of final-period (simple), compound final-period and long bonds """
    long_bond = bonds.bond_flows(sch.make_bond_scheduler(dt.date(2025, 11, 15), dt.date(2035, 11, 15), UST),
                                 SETTLE, VALDATEOLD)
    flows, coupons = [final_period_note(), final_period_note(COMPOUND), long_bond], [0.0425, 0.0425, 0.04]
    ff = bonds.flatten(flows, coupons)
    assert ff.simple.tolist() == [True, False] + [False] * len(long_bond.periods)
    y = np.array([0.04, 0.041, 0.043])
    dirty = bonds.dirty_from_yields(ff, y)
    np.testing.assert_allclose(dirty, [bonds.dirty_from_yield(bf, c, yi) for bf, c, yi in zip(flows, coupons, y, strict=True)],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(bonds.yields_from_dirty(ff, dirty), y, rtol=0, atol=1e-13)
    np.testing.assert_allclose(bonds.dv01s(ff, y), [bonds.dv01(bf, c, yi) for bf, c, yi in zip(flows, coupons, y, strict=True)],
                               rtol=1e-13)
