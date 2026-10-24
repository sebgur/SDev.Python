import datetime as dt
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sdevpy.utilities.scalendar import BDC, Period, make_calendar
from sdevpy.utilities.sdaycount import DayCount, yearfraction
from sdevpy.utilities import timegrids

D = 'datetime64[D]'


@dataclass(frozen=True)
class LegConvention:
    frequency: str
    daycount: DayCount
    calendar: str  # name understood by make_calendar()
    bdc: BDC = BDC.MF
    adjust_accrual: bool = True
    pay_lag: int = 0  # business days after adjusted accrual end
    fixing_lag: int | None = None  # business days before accrual start
    fixing_calendar: str | None = None  # index fixing calendar (None: use the leg calendar)
    eom: bool = False


@dataclass(frozen=True)
class LegSchedule:
    """ All arrays have shape (n_periods,), dates as datetime64[D] """
    unadj_start: npt.NDArray[np.datetime64]
    unadj_end: npt.NDArray[np.datetime64]
    adj_start: npt.NDArray[np.datetime64]
    adj_end: npt.NDArray[np.datetime64]
    payment: npt.NDArray[np.datetime64]
    accrual_fraction: npt.NDArray[np.float64]
    fixing: npt.NDArray[np.datetime64] | None = None

    def payment_times(self, valdate: dt.date) -> npt.NDArray[np.float64]:
        return timegrids.model_time(valdate, self.payment)

    @property
    def n_periods(self) -> int:
        return self.payment.shape[-1]


def leg_from_periods(periods: list[Period], conv: LegConvention) -> LegSchedule:
    """ Create LegSchedule from list of periods """
    cal = make_calendar(conv.calendar)
    unadj_start = np.array([p.unadj_start for p in periods], dtype=D)
    unadj_end = np.array([p.unadj_end for p in periods], dtype=D)
    adj_start = np.array([p.adj_start for p in periods], dtype=D)
    adj_end = np.array([p.adj_end for p in periods], dtype=D)

    payment = adj_end
    if conv.pay_lag != 0:
        payment = np.array([cal.add_business_days(p.adj_end, conv.pay_lag) for p in periods], dtype=D)

    fixing = None
    if conv.fixing_lag is not None:
        fix_cal = make_calendar(conv.fixing_calendar or conv.calendar)
        fixing = np.array([fix_cal.add_business_days(p.adj_start, -conv.fixing_lag) for p in periods], dtype=D)

    if conv.adjust_accrual:
        alpha = [yearfraction(p.adj_start, p.adj_end, conv.daycount) for p in periods]
    else:
        alpha = [yearfraction(p.unadj_start, p.unadj_end, conv.daycount) for p in periods]

    return LegSchedule(unadj_start, unadj_end, adj_start, adj_end, payment,
                       np.array(alpha, dtype=np.float64), fixing)


def make_leg(start: dt.date, end: dt.date, conv: LegConvention, **kwargs) -> LegSchedule:
    """ kwargs go to Calendar.make_periods: stub, roll_convention, first_date, next_to_last_date,
        termination_convention """
    periods = make_calendar(conv.calendar).make_periods(start, end, conv.frequency, convention=conv.bdc,
                                                         eom=conv.eom, **kwargs)
    return leg_from_periods(periods, conv)
