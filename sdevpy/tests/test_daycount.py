import datetime as dt
# from sdevpy.utilities import scalendar as cdr
from sdevpy.utilities import sdaycount as dc


def test_yearfraction_act360():
    start, end = dt.date(2024, 1, 15), dt.date(2024, 4, 15)
    test = dc.yearfraction(start, end, dc.DayCount.ACT360)
    ref = 91 / 360
    assert abs(test - ref) < 1e-12


def test_yearfraction_act365f():
    start, end = dt.date(2024, 1, 15), dt.date(2024, 4, 15)
    test = dc.yearfraction(start, end, dc.DayCount.ACT365F)
    ref = 91 / 365
    assert abs(test - ref) < 1e-12


def test_yearfraction_30_360():
    # 30/360 treats each month as 30 days regardless of actual length
    start, end = dt.date(2024, 1, 31), dt.date(2024, 2, 29)
    test = dc.yearfraction(start, end, dc.DayCount.D30_360)
    ref = 29 / 360  # d1 clamped 31->30, d2=29 (not 31, no clamp) -> 0*360 + 1*30 + (29-30)
    assert abs(test - ref) < 1e-12
