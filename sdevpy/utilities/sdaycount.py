import datetime as dt
from enum import Enum
from sdevpy.utilities.scalendar import Period


class DayCount(Enum):
    ACT360 = "ACT/360"
    ACT365F = "ACT/365F"
    D30_360 = "30/360"
    D30E_360 = "30E/360"


def year_fraction(start: dt.date, end: dt.date, convention: DayCount) -> float:
    """ Year fraction between two dates under the given day-count convention """
    if convention == DayCount.ACT360:
        return (end - start).days / 360
    if convention == DayCount.ACT365F:
        return (end - start).days / 365
    if convention == DayCount.D30_360:
        return _30_360(start, end)
    if convention == DayCount.D30E_360:
        return _30e_360(start, end)
    raise ValueError(f"Unsupported day count: {convention}")


def _30_360(start: dt.date, end: dt.date) -> float:
    """ US (NASD) 30/360 """
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return ((end.year - start.year) * 360
            + (end.month - start.month) * 30
            + (d2 - d1)) / 360


def _30e_360(start: dt.date, end: dt.date) -> float:
    """ 30E/360 (Eurobond) """
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    return ((end.year - start.year) * 360
            + (end.month - start.month) * 30
            + (d2 - d1)) / 360


# ACT/360 and ACT/365F are defined on the realized (adjusted) period;
# 30/360 and 30E/360 are defined on the unadjusted schedule dates.
_USES_UNADJUSTED = {DayCount.D30_360, DayCount.D30E_360}


def period_year_fraction(period: Period, convention: DayCount) -> float:
    """ Year fraction for one accrual period, picking the correct date pair for the convention """
    if convention in _USES_UNADJUSTED:
        return year_fraction(period.unadj_start, period.unadj_end, convention)
    return year_fraction(period.adj_start, period.adj_end, convention)
