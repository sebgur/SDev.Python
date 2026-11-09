""" Static conventions for rates instruments, as stored in the convention repository (see repository.py).
    All objects are immutable and refer to each other by id (e.g. a swap refers to its float index by id) """
from dataclasses import dataclass
from enum import Enum
from sdevpy.utilities.calendars import BDC
from sdevpy.utilities.daycounts import DayCount


class IndexKind(Enum):
    OVERNIGHT = "overnight"  # e.g. SOFR, ESTR, SONIA: compounded/averaged over the accrual period
    TERM = "term"  # e.g. EURIBOR: fixed once per period


@dataclass(frozen=True)
class IndexConvention:
    id: str
    notes: str
    currency: str
    kind: IndexKind
    tenor: str | None  # None for overnight indices
    daycount: DayCount
    calendar: str  # fixing calendar
    fixing_lag: int  # business days from fixing to value date
    publication_lag: int  # business days from value date to publication (overnight indices)
    bdc: BDC  # maturity adjustment of a term index
    eom: bool
