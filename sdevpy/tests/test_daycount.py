import datetime as dt
from sdevpy.utilities import scalendar as cdr
from sdevpy.utilities import sdaycount as dc


def test_year_fraction_act360():
    start, end = dt.date(2024, 1, 15), dt.date(2024, 4, 15)
    test = dc.year_fraction(start, end, dc.DayCount.ACT360)
    ref = 91 / 360
    assert abs(test - ref) < 1e-12


def test_year_fraction_act365f():
    start, end = dt.date(2024, 1, 15), dt.date(2024, 4, 15)
    test = dc.year_fraction(start, end, dc.DayCount.ACT365F)
    ref = 91 / 365
    assert abs(test - ref) < 1e-12


def test_year_fraction_30_360():
    # 30/360 treats each month as 30 days regardless of actual length
    start, end = dt.date(2024, 1, 31), dt.date(2024, 2, 29)
    test = dc.year_fraction(start, end, dc.DayCount.D30_360)
    ref = 29 / 360  # d1 clamped 31->30, d2=29 (not 31, no clamp) -> 0*360 + 1*30 + (29-30)
    assert abs(test - ref) < 1e-12


def test_period_year_fraction_uses_correct_dates():
    cal1 = cdr.make_calendar("USD")
    cal2 = cdr.make_calendar("NYSE")
    cal = cal1 + cal2

    start = dt.date(2024, 1, 15)
    end = dt.date(2025, 1, 15)
    periods = cal.make_periods(start, end, '3M')

    # period[0] unadj_start = 2024-01-15 (MLK), adj_start = 2024-01-16
    act360 = dc.period_year_fraction(periods[0], dc.DayCount.ACT360)
    ref_act360 = (periods[0].adj_end - periods[0].adj_start).days / 360
    assert abs(act360 - ref_act360) < 1e-12

    d30_360 = dc.period_year_fraction(periods[0], dc.DayCount.D30_360)
    ref_30_360 = dc.year_fraction(periods[0].unadj_start, periods[0].unadj_end, dc.DayCount.D30_360)
    assert abs(d30_360 - ref_30_360) < 1e-12
