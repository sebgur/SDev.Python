import datetime as dt
import numpy as np
import dataclasses
import pytest
from sdevpy.utilities.schedules import make_bond_scheduler, BondConvention
from sdevpy.rates.bonds import bond_flows, accrued_interest, dirty_from_curve, yield_from_dirty
from sdevpy.calibration.rates import yieldcurve as ycrv
from sdevpy.rates.bondquotes import (settlement_date, bill_flows, bill_days, price_from_discount_rate,
                                     discount_rate_from_price, QuoteType, BondQuote, target_dirty, fit_weight,
                                     parse_32nds)
from sdevpy.conventions.bondconventions import UST, GILT


VALDATE = dt.datetime(2026, 10, 26)
# UST = LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)


def test_settlement_skips_holiday():
    # Wed 25-Nov-2026 + 1 skips Thanksgiving (Thu 26-Nov)
    assert settlement_date(dt.date(2026, 11, 25), UST) == dt.date(2026, 11, 27)
    assert settlement_date(dt.datetime(2026, 10, 23), UST) == dt.date(2026, 10, 26)


def test_bill_discount_rate():
    bill = bill_flows(dt.date(2027, 1, 26), dt.date(2026, 10, 27))
    assert bill_days(bill) == 91
    p = price_from_discount_rate(bill, 0.04)
    assert p == pytest.approx(100.0 * (1.0 - 0.04 * 91 / 360), abs=1e-12)
    assert discount_rate_from_price(bill, p) == pytest.approx(0.04, abs=1e-14)


def test_bill_matured():
    with pytest.raises(ValueError):
        bill_flows(dt.date(2026, 10, 27), dt.date(2026, 10, 27))


def test_bill_curve_price():
    curve = ycrv.InterpolatedYieldCurve(valdate=VALDATE, interp_var='zerorate', interp_type='linear')
    curve.set_data([dt.datetime(2027, 10, 26), dt.datetime(2028, 10, 26)], [np.exp(-0.04), np.exp(-0.08)])
    bill = bill_flows(dt.date(2027, 1, 26), dt.date(2026, 10, 27), VALDATE)
    assert dirty_from_curve(bill, 0.0, curve.discount_float) == pytest.approx(100.0 * np.exp(-0.04 * 91 / 365))


@pytest.mark.parametrize("s, expected", [('99-16', 99.5), ('99-16+', 99.515625), ('99-162', 99.5078125),
                                         ('101-07', 101.21875), ('100', 100.0), (' 98-31+ ', 98 + 31.5 / 32)])
def test_parse_32nds(s, expected):
    assert parse_32nds(s) == pytest.approx(expected, abs=1e-14)


def test_quote_types_consistent():
    settle = settlement_date(VALDATE, UST)
    note = bond_flows(make_bond_scheduler(dt.date(2026, 8, 15), dt.date(2036, 8, 15), UST), settle, VALDATE)
    cpn, clean = 0.04125, 99.515625
    dirty = clean + accrued_interest(note, cpn)
    qd = BondQuote(note, cpn, QuoteType.DIRTY, dirty)
    w = fit_weight(qd, dirty)
    from sdevpy.rates.bonds import yield_from_dirty
    ytm = yield_from_dirty(note, cpn, dirty)
    for qt, v in [(QuoteType.CLEAN, clean), (QuoteType.YIELD, ytm)]:
        q = BondQuote(note, cpn, qt, v)
        assert target_dirty(q) == pytest.approx(dirty, abs=1e-10)
        assert fit_weight(q, target_dirty(q)) == pytest.approx(w, rel=1e-10)


def test_weight_is_inverse_dv01():
    # 1bp move in yield moves the dirty price by ~1/weight
    settle = settlement_date(VALDATE, UST)
    note = bond_flows(make_bond_scheduler(dt.date(2026, 8, 15), dt.date(2036, 8, 15), UST), settle, VALDATE)
    q = BondQuote(note, 0.04, QuoteType.YIELD, 0.042)
    q_up = BondQuote(note, 0.04, QuoteType.YIELD, 0.042 + 1e-4)
    dp = target_dirty(q) - target_dirty(q_up)
    assert dp * fit_weight(q, target_dirty(q)) == pytest.approx(1.0, abs=5e-3)


def test_settlement_usgov_good_friday():
    # Thu 17-Apr-2025 + 1: Good Friday is a SIFMA holiday but not a federal one
    assert settlement_date(dt.date(2025, 4, 17), UST) == dt.date(2025, 4, 21)
    federal = BondConvention(dataclasses.replace(UST.leg, calendar='USD'))
    assert settlement_date(dt.date(2025, 4, 17), federal) == dt.date(2025, 4, 18)


def test_gilt_settlement_london_holidays():
    # Thu 2-Apr-2026 + 1: Good Friday 3-Apr and Easter Monday 6-Apr closed; Fri 28-Aug-2026 + 1: bank holiday 31-Aug
    assert settlement_date(dt.date(2026, 4, 2), GILT) == dt.date(2026, 4, 7)
    assert settlement_date(dt.date(2026, 8, 28), GILT) == dt.date(2026, 9, 1)


def test_bill_bond_equivalent_yield():
    """ Bills up to 182 days: yield = Treasury bond-equivalent (investment) rate (100/P - 1) 365/days """
    bill = bill_flows(dt.date(2026, 3, 17), dt.date(2025, 12, 16))  # 13 weeks, 91 days
    p = price_from_discount_rate(bill, 0.04)
    assert yield_from_dirty(bill, 0.0, p) == pytest.approx((100 / p - 1) * 365 / 91, abs=1e-13)
