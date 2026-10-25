import datetime as dt
import numpy as np
import pytest
from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount, icma_yearfraction, yearfraction
from sdevpy.utilities import schedules as sch
from sdevpy.conventions.bondconventions import UST, GILT


# GILT = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'GBP', bdc=BDC.F, adjust_accrual=False)
# UST = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
SOFR = sch.LegConvention('1Y', DayCount.ACT360, 'USD', pay_lag=2)
USD_OIS = sch.SwapConvention(SOFR, SOFR)


def dates64(*dates):
    return np.array(dates, dtype='datetime64[D]')


################ ICMA ##############################################################################

def test_icma_isda_examples():
    d = dt.date
    assert abs(icma_yearfraction(d(1999, 2, 1), d(1999, 7, 1), [d(1998, 7, 1), d(1999, 7, 1)], 1) - 150 / 365) < 1e-12
    assert abs(icma_yearfraction(d(2002, 8, 15), d(2003, 7, 15), [d(2002, 7, 15), d(2003, 7, 15)], 1) - 334 / 365) < 1e-12


def test_icma_bond_stubs():
    d = dt.date
    short = sch.make_bond_scheduler(d(2026, 1, 10), d(2031, 2, 15), GILT).schedule.accrual_fraction
    long = sch.make_bond_scheduler(d(2026, 1, 10), d(2031, 2, 15), GILT, stub='long_front').schedule.accrual_fraction
    back = sch.make_bond_scheduler(d(2026, 2, 15), d(2030, 10, 1), GILT, stub='short_back').schedule.accrual_fraction
    assert np.allclose(short, [36 / 368] + [0.5] * 10, rtol=0, atol=1e-12)
    assert np.allclose(long, [36 / 368 + 0.5] + [0.5] * 9, rtol=0, atol=1e-12)
    assert np.allclose(back, [0.5] * 9 + [47 / 368], rtol=0, atol=1e-12)


def test_icma_needs_schedule():
    with pytest.raises(ValueError):
        yearfraction(dt.date(2026, 1, 1), dt.date(2026, 7, 1), DayCount.ACTACT_ICMA)


def test_icma_explicit_first_coupon():
    d = dt.date
    alpha = sch.make_bond_scheduler(d(2026, 1, 10), d(2031, 2, 15), GILT, first_coupon=d(2026, 8, 15)).schedule.accrual_fraction
    assert np.allclose(alpha, [36 / 368 + 0.5] + [0.5] * 9, rtol=0, atol=1e-12)


def test_quasi_coupon_dates():
    d = dt.date
    # Short stub inside the regular half-year 15-Aug-25 -> 15-Feb-26
    assert sch.quasi_coupon_dates(d(2031, 2, 15), d(2026, 1, 10), d(2026, 2, 15), '6M') == [d(2025, 8, 15), d(2026, 2, 15)]
    # Long stub spanning two regular half-years
    assert sch.quasi_coupon_dates(d(2031, 2, 15), d(2026, 1, 10), d(2026, 8, 15), '6M') == \
        [d(2025, 8, 15), d(2026, 2, 15), d(2026, 8, 15)]
    # Anchor before the period (back stub), and end-of-month rule
    assert sch.quasi_coupon_dates(d(2026, 2, 15), d(2030, 8, 15), d(2030, 10, 1), '6M') == [d(2030, 8, 15), d(2031, 2, 15)]
    assert sch.quasi_coupon_dates(d(2035, 11, 30), d(2026, 5, 31), d(2026, 11, 30), '6M', eom=True) == \
        [d(2026, 5, 31), d(2026, 11, 30)]


def test_regular_anchor():
    d = dt.date
    start, end, first, ntl = d(2026, 1, 10), d(2031, 2, 15), d(2026, 8, 15), d(2030, 8, 15)
    assert sch.regular_anchor(start, end) == end
    assert sch.regular_anchor(start, end, stub='long_front') == end
    assert sch.regular_anchor(start, end, next_to_last_date=ntl) == ntl
    assert sch.regular_anchor(start, end, stub='short_back') == start
    assert sch.regular_anchor(start, end, stub='long_back', first_date=first) == first


def test_coupons_per_year():
    assert [sch.coupons_per_year(f) for f in ('1M', '3M', '6M', '1Y')] == [12, 4, 2, 1]
    for f in ('5M', '2Y', '1W'):
        with pytest.raises(ValueError):
            sch.coupons_per_year(f)


################ Legs ##############################################################################

def test_leg_payment_lag_and_accruals():
    leg = sch.make_leg_schedule(dt.date(2026, 10, 26), dt.date(2031, 10, 26), SOFR)
    assert (leg.adj_end == dates64('2027-10-26', '2028-10-26', '2029-10-26', '2030-10-28', '2031-10-27')).all()
    assert (leg.payment == dates64('2027-10-28', '2028-10-30', '2029-10-30', '2030-10-30', '2031-10-29')).all()
    ref = [365 / 360, 366 / 360, 365 / 360, 367 / 360, 364 / 360]
    assert np.allclose(leg.accrual_fraction, ref, rtol=0, atol=1e-12)
    assert leg.fixing is None
    assert leg.n_periods == 5


def test_leg_fixing_calendar():
    # Period starting Wed 5-May-2027, two days after the UK May bank holiday (TARGET open)
    on_leg_cal = sch.LegConvention('6M', DayCount.ACT360, 'EUR,GBP', fixing_lag=2)
    on_target = sch.LegConvention('6M', DayCount.ACT360, 'EUR,GBP', fixing_lag=2, fixing_calendar='EUR')
    assert sch.make_leg_schedule(dt.date(2027, 5, 5), dt.date(2028, 5, 5), on_leg_cal).fixing[0] == np.datetime64('2027-04-30')
    assert sch.make_leg_schedule(dt.date(2027, 5, 5), dt.date(2028, 5, 5), on_target).fixing[0] == np.datetime64('2027-05-03')


def test_leg_bond_style_accrual():
    # Payments roll to business days, accruals stay on the unadjusted dates
    bond_style = sch.LegConvention('6M', DayCount.D30_360, 'USD', bdc=BDC.F, adjust_accrual=False)
    swap_style = sch.LegConvention('6M', DayCount.D30_360, 'USD', bdc=BDC.F, adjust_accrual=True)
    start, end = dt.date(2026, 2, 15), dt.date(2030, 2, 15)
    leg = sch.make_leg_schedule(start, end, bond_style)
    assert (leg.payment[:2] == dates64('2026-08-17', '2027-02-16')).all()  # weekend, Presidents' Day
    assert np.allclose(leg.accrual_fraction, 0.5, rtol=0, atol=1e-12)
    adj = sch.make_leg_schedule(start, end, swap_style).accrual_fraction
    assert abs(adj[1] - 179 / 360) < 1e-12  # adjusted 17-Aug-26 -> 16-Feb-27 on 30/360


def test_payment_times():
    leg = sch.make_leg_schedule(dt.date(2026, 10, 26), dt.date(2028, 10, 26), SOFR)
    t = leg.payment_times(dt.datetime(2026, 10, 22))
    assert np.allclose(t, [371 / 365, 739 / 365], rtol=0, atol=1e-12)


################ Swaps #############################################################################

def test_spot_swap_dates():
    swap = sch.make_spot_swap_scheduler(dt.date(2026, 10, 23), '3Y', USD_OIS)  # Friday trade
    assert swap.start == np.datetime64('2026-10-27')  # T+2 = Tuesday
    assert swap.maturity == np.datetime64('2029-10-29')  # 27-Oct-2029 is a Saturday
    assert swap.fixed.n_periods == 3 and swap.floating.n_periods == 3


def test_forward_start_swap():
    swap = sch.make_spot_swap_scheduler(dt.date(2026, 10, 23), '5Y', USD_OIS, forward='5Y')
    assert swap.start == np.datetime64('2031-10-27')
    assert swap.maturity == np.datetime64('2036-10-27')


def test_swap_legs_with_different_frequencies():
    eur = sch.SwapConvention(sch.LegConvention('1Y', DayCount.D30_360, 'EUR'),
                             sch.LegConvention('6M', DayCount.ACT360, 'EUR', fixing_lag=2))
    swap = sch.make_spot_swap_scheduler(dt.date(2026, 10, 23), '2Y', eur)
    assert swap.fixed.n_periods == 2 and swap.floating.n_periods == 4
    assert swap.floating.fixing[0] == np.datetime64('2026-10-23')  # T-2 from spot = trade date
    assert swap.fixed.adj_end[-1] == swap.floating.adj_end[-1]


################ Bonds #############################################################################

def test_bond_schedule():
    bond = sch.make_bond_scheduler(dt.date(2025, 11, 30), dt.date(2035, 11, 30), UST)
    assert (bond.schedule.unadj_end[:3] == dates64('2026-05-31', '2026-11-30', '2027-05-31')).all()  # EOM rule
    assert bond.schedule.payment[0] == np.datetime64('2026-06-01')  # 31-May-2026 is a Sunday
    assert bond.maturity == np.datetime64('2035-11-30')
    assert bond.coupons_per_year == 2
    assert bond.anchor == dt.date(2035, 11, 30)
    assert np.allclose(bond.schedule.accrual_fraction, 0.5, rtol=0, atol=1e-12)


def test_bond_maturity_on_business_day():
    bond = sch.make_bond_scheduler(dt.date(2026, 1, 10), dt.date(2031, 2, 15), GILT)  # 15-Feb-2031 is a Saturday
    assert bond.maturity == np.datetime64('2031-02-17')


def test_bond_accrual_fraction():
    d = dt.date
    bond = sch.make_bond_scheduler(d(2026, 2, 15), d(2036, 2, 15), UST)
    assert abs(bond.accrual_fraction(d(2026, 2, 15), d(2026, 5, 15)) - 0.5 * 89 / 181) < 1e-12
    stub_bond = sch.make_bond_scheduler(d(2026, 1, 10), d(2031, 2, 15), GILT)
    assert abs(stub_bond.accrual_fraction(d(2026, 1, 10), d(2026, 1, 20)) - 10 / 368) < 1e-12


def test_bond_datetime_inputs():
    a = sch.make_bond_scheduler(dt.date(2026, 1, 10), dt.date(2031, 2, 15), GILT)
    b = sch.make_bond_scheduler(dt.datetime(2026, 1, 10), dt.datetime(2031, 2, 15), GILT)
    assert (a.schedule.payment == b.schedule.payment).all()
    assert np.allclose(a.schedule.accrual_fraction, b.schedule.accrual_fraction, rtol=0, atol=1e-15)
    assert b.anchor == dt.date(2031, 2, 15)


def test_to_date():
    d = dt.date(2026, 5, 15)
    assert sch.to_date(d) == d
    assert sch.to_date(dt.datetime(2026, 5, 15, 10, 30)) == d
    assert sch.to_date(np.datetime64('2026-05-15')) == d
    assert type(sch.to_date(np.datetime64('2026-05-15'))) is dt.date
