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


def yearfraction(start: dt.date, end: dt.date, convention: DayCount) -> float|npt.NDArray[np.float64]:
    """ Year fraction between two dates, or two broadcastable arrays/lists of dates, under the given
        day-count convention.
        ACT360 and ACT365F: fully vectorized via numpy datetime64 arithmetic.
        D30_360 and D30E_360: scalar-only. These index day/month/year components per date rather than
                              a single elapsed day count, so vectorizing them is separate work. """
    if convention in (DayCount.ACT360, DayCount.ACT365F):
        days = (np.asarray(end, dtype='datetime64[D]')
                - np.asarray(start, dtype='datetime64[D]')).astype(np.float64)
        denom = 360.0 if convention == DayCount.ACT360 else 365.0
        result = days / denom
        return float(result) if result.ndim == 0 else result

    if convention == DayCount.D30_360:
        return _30_360(start, end)

    if convention == DayCount.D30E_360:
        return _30e_360(start, end)

    if convention == DayCount.ACTACT_ICMA:
        raise ValueError("ACT/ACT ICMA needs a coupon schedule: use icma_yearfraction()")

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


# def period_yearfraction(period: Period, convention: DayCount) -> float:
#     """ Year fraction for one accrual period, picking the correct date pair for the convention """
#     return yearfraction(period.adj_start, period.adj_end, convention)
