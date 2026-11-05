import datetime as dt
from enum import Enum
import numpy as np
import numpy.typing as npt
# from sdevpy.utilities.scalendar import Period


class DayCount(Enum):
    ACT360 = "ACT/360"
    ACT365F = "ACT/365F"
    D30_360 = "30/360"
    D30E_360 = "30E/360"
    ACTACT_ICMA = "ACT/ACT ICMA"
    ACT365_NL = "NL/365" # actual days with Feb 29 not counted / 365 (JGB accrued and simple yield)
    ACT365_CANADIAN = "ACT/365 Canadian bond"  # needs the coupon period: act365_canadian()


def yearfraction(start: dt.date, end: dt.date, convention: DayCount) -> float|npt.NDArray[np.float64]:
    """ Year fraction between two dates, or two broadcastable arrays/lists of dates, under the given
        day-count convention.
        ACT360, ACT365F, ACT365_NL: fully vectorized via numpy datetime64 arithmetic.
        D30_360 and D30E_360: scalar-only. These index day/month/year components per date rather than
                              a single elapsed day count, so vectorizing them is separate work. """
    if convention in (DayCount.ACT360, DayCount.ACT365F):
        days = (np.asarray(end, dtype='datetime64[D]')
                - np.asarray(start, dtype='datetime64[D]')).astype(np.float64)
        denom = 360.0 if convention == DayCount.ACT360 else 365.0
        result = days / denom
        return float(result) if result.ndim == 0 else result

    if convention == DayCount.ACT365_NL:
        result = _nl_days(start, end) / 365.0
        return float(result) if result.ndim == 0 else result

    if convention == DayCount.D30_360:
        return _30_360(start, end)

    if convention == DayCount.D30E_360:
        return _30e_360(start, end)

    if convention in (DayCount.ACTACT_ICMA, DayCount.ACT365_CANADIAN):
        raise ValueError(f"{convention.value} needs a coupon schedule")

    raise ValueError(f"Unsupported day count: {convention}")


def icma_yearfraction(start: dt.date, end: dt.date, quasi_dates: list[dt.date], frequency: int) -> float:
    """ ACT/ACT ICMA (ICMA Rule 251). quasi_dates are the regular (quasi-)coupon dates covering [start, end].
        Sum over quasi-coupon periods (q0, q1) of the days of [start, end] inside (q0, q1), divided by
        frequency * days in (q0, q1). A full regular period gives exactly 1/frequency. """
    total = 0.0
    for q0, q1 in zip(quasi_dates[:-1], quasi_dates[1:], strict=True):
        overlap = (min(end, q1) - max(start, q0)).days
        if overlap > 0:
            total += overlap / (frequency * (q1 - q0).days)
    return total


def act365_canadian(start: dt.date, end: dt.date, period_end: dt.date, frequency: int) -> float:
    """ Actual/365 (Canadian bond), accrued interest on Government of Canada bonds: days/365 for fewer than 365/f
        days (182 semi-annual) into the period, then 1/f - (days left to period_end)/365, reaching 1/f exactly """
    days = (end - start).days
    if days < 365 // frequency:
        return days / 365.0
    return 1.0 / frequency - (period_end - end).days / 365.0


def _nl_days(start, end) -> npt.NDArray[np.float64]:
    """ Actual days in (start, end], Feb 29 not counted """
    s, e = np.asarray(start, dtype='datetime64[D]'), np.asarray(end, dtype='datetime64[D]')
    return (e - s).astype(np.float64) - (_leap_days_through(e) - _leap_days_through(s))


def _leap_days_through(d: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int64]:
    """ Number of Feb 29 on or before each date, counted from year 1 """
    y = d.astype('datetime64[Y]').astype(np.int64) + 1970
    p = y - 1
    leap = (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))
    doy = (d - d.astype('datetime64[Y]')).astype(np.int64)  # 0-based day of year, Feb 29 = 59
    return p // 4 - p // 100 + p // 400 + (leap & (doy >= 59))


def _30_360(start: dt.date, end: dt.date) -> float:
    """ US (NASD) 30/360 """
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360


def _30e_360(start: dt.date, end: dt.date) -> float:
    """ 30E/360 (Eurobond) """
    d1 = min(start.day, 30)
    d2 = min(end.day, 30)
    return ((end.year - start.year) * 360 + (end.month - start.month) * 30 + (d2 - d1)) / 360
