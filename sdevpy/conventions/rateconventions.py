""" Static conventions for rates instruments, as stored in the convention repository (see repository.py).
    All objects are immutable and refer to each other by id (e.g. a swap refers to its float index by id) """
from dataclasses import dataclass
from enum import Enum
from sdevpy.utilities.calendars import BDC
from sdevpy.utilities.daycounts import DayCount
from sdevpy.utilities.schedules import LegConvention, SwapConvention


class IndexKind(Enum):
    OVERNIGHT = "overnight"  # e.g. SOFR, ESTR, SONIA: compounded/averaged over the accrual period
    TERM = "term"  # e.g. EURIBOR: fixed once per period


class SwapKind(Enum):
    IRS = "irs"  # fixed vs term index
    OIS = "ois"  # fixed vs compounded overnight index


class RateMethod(Enum):
    COMPOUNDED = "compounded"  # overnight index compounded over the reference period (e.g. CME SR3)
    AVERAGED = "averaged"  # overnight index arithmetic average over the reference period (e.g. CME SR1)
    TERM = "term"  # single term index fixing (e.g. EURIBOR futures)


class FutureDateRule(Enum):
    IMM = "imm"  # reference period starts on the 3rd Wednesday of the contract month
    MONTH = "month"  # reference period is the contract calendar month


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


@dataclass(frozen=True)
class DepositConvention:
    id: str
    notes: str
    currency: str
    daycount: DayCount
    calendar: str
    spot_lag: int  # business days from trade date to start
    bdc: BDC
    eom: bool


@dataclass(frozen=True)
class FraConvention:
    id: str
    notes: str
    currency: str
    index: str  # term index id: daycount, calendar, fixing lag and maturity rules come from the index
    spot_lag: int  # business days from trade date to spot, from which the FRA start is counted


@dataclass(frozen=True)
class FutureConvention:
    id: str
    notes: str
    currency: str
    index: str  # index id
    rate_method: RateMethod
    period: str  # length of the reference period, e.g. '3M'
    date_rule: FutureDateRule
    exchange: str
    notional: float  # contract notional in currency


@dataclass(frozen=True)
class FixFloatSwapConvention:
    id: str
    notes: str
    kind: SwapKind
    currency: str
    float_index: str  # index id
    fixed_leg: LegConvention
    float_leg: LegConvention
    spot_lag: int  # business days from trade date to start
    spot_calendar: str | None  # None: fixed leg calendar

    @property
    def schedule_convention(self) -> SwapConvention:
        """ Convention consumed by the schedulers, e.g. make_spot_swap_scheduler """
        return SwapConvention(self.fixed_leg, self.float_leg, self.spot_lag, self.spot_calendar)
