import datetime as dt
import numpy as np
from sdevpy.utilities.scalendar import BDC
from sdevpy.utilities.sdaycount import DayCount, icma_yearfraction
from sdevpy.utilities import schedules as sch

GILT = sch.LegConvention('6M', DayCount.ACTACT_ICMA, 'GBP', bdc=BDC.F, adjust_accrual=False)


def test_icma_isda_examples():
    d = dt.date
    assert abs(icma_yearfraction(d(1999, 2, 1), d(1999, 7, 1), [d(1998, 7, 1), d(1999, 7, 1)], 1) - 150 / 365) < 1e-12
    assert abs(icma_yearfraction(d(2002, 8, 15), d(2003, 7, 15), [d(2002, 7, 15), d(2003, 7, 15)], 1) - 334 / 365) < 1e-12


def test_icma_bond_stubs():
    d = dt.date
    short = sch.make_bond(d(2026, 1, 10), d(2031, 2, 15), GILT).coupons.accrual_fraction
    long = sch.make_bond(d(2026, 1, 10), d(2031, 2, 15), GILT, stub='long_front').coupons.accrual_fraction
    back = sch.make_bond(d(2026, 2, 15), d(2030, 10, 1), GILT, stub='short_back').coupons.accrual_fraction
    assert np.allclose(short, [36 / 368] + [0.5] * 10, rtol=0, atol=1e-12)
    assert np.allclose(long, [36 / 368 + 0.5] + [0.5] * 9, rtol=0, atol=1e-12)
    assert np.allclose(back, [0.5] * 9 + [47 / 368], rtol=0, atol=1e-12)
