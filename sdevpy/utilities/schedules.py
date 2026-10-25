import datetime as dt
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from sdevpy.utilities.scalendar import BDC, Period, make_calendar, is_eom, to_eom
from sdevpy.utilities.sdaycount import DayCount, yearfraction, icma_yearfraction
from sdevpy.utilities import timegrids
from sdevpy.utilities import dates as dts


D = 'datetime64[D]'


################### LEG ###########################################################################
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


def coupons_per_year(frequency: str) -> int:
    p = dts.period(frequency)
    months = 12 * p.years + p.months
    if p.days != 0 or months <= 0 or 12 % months != 0:
        raise ValueError(f"Frequency '{frequency}' is not a whole number of periods per year")
    return 12 // months


def quasi_coupon_dates(anchor: dt.date, start: dt.date, end: dt.date, frequency: str,
                       eom: bool = False) -> list[dt.date]:
    """ Regular coupon dates anchor + k * frequency (k of either sign) covering [start, end], i.e. from the last
        one on or before start to the first one on or after end. Computed from the anchor (no roll-day drift). """
    period = dts.period(frequency)
    use_eom = eom and is_eom(anchor)

    def nth(k: int) -> dt.date:
        d = anchor + period * k
        return to_eom(d) if use_eom else d

    k = 0
    while nth(k) > start:  # step back to the last date on or before start
        k -= 1
    while nth(k + 1) <= start:  # or forward, if the anchor is before start
        k += 1
    dates = [nth(k)]
    while dates[-1] < end:
        k += 1
        dates.append(nth(k))
    return dates


def regular_anchor(start: dt.date, end: dt.date, stub: str = "short_front", first_date: dt.date | None = None,
                   next_to_last_date: dt.date | None = None, **kwargs) -> dt.date:
    """ Date the regular part of the schedule is rolled from (same logic as Calendar._unadjusted_roll_dates) """
    if stub in ("short_back", "long_back"):
        return first_date or start
    return next_to_last_date or end


def accrual_fractions(starts: list[dt.date], ends: list[dt.date], conv: LegConvention,
                      anchor: dt.date) -> list[float]:
    """ Accrual fraction per period. For ACT/ACT ICMA, only the first and last periods can be irregular: they are
        measured against quasi-coupon periods rolled from the schedule's regular anchor. """
    if conv.daycount != DayCount.ACTACT_ICMA:
        return [yearfraction(s, e, conv.daycount) for s, e in zip(starts, ends, strict=True)]

    freq = coupons_per_year(conv.frequency)
    alpha = []
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        if 0 < i < len(starts) - 1:
            quasi = [s, e]  # regular by construction
        else:
            quasi = quasi_coupon_dates(anchor, s, e, conv.frequency, conv.eom)
        alpha.append(icma_yearfraction(s, e, quasi, freq))
    return alpha


def leg_from_periods(periods: list[Period], conv: LegConvention, anchor: dt.date|None=None) -> LegSchedule:
    """ Create LegSchedule from list of periods. anchor: date the regular periods are rolled from, only used by
        ACT/ACT ICMA (None: end of the last period, i.e. backward-generated schedule) """
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
        starts, ends = [p.adj_start for p in periods], [p.adj_end for p in periods]
    else:
        starts, ends = [p.unadj_start for p in periods], [p.unadj_end for p in periods]
    anchor = anchor or periods[-1].unadj_end
    alpha = accrual_fractions(starts, ends, conv, anchor)

    return LegSchedule(unadj_start, unadj_end, adj_start, adj_end, payment,
                       np.array(alpha, dtype=np.float64), fixing)


def make_leg_schedule(start: dt.date, end: dt.date, conv: LegConvention, **kwargs) -> LegSchedule:
    """ kwargs go to Calendar.make_periods: stub, roll_convention, first_date, next_to_last_date,
        termination_convention """
    periods = make_calendar(conv.calendar).make_periods(start, end, conv.frequency, convention=conv.bdc,
                                                         eom=conv.eom, **kwargs)
    return leg_from_periods(periods, conv, regular_anchor(start, end, **kwargs))


################## SWAP ###########################################################################
@dataclass(frozen=True)
class SwapConvention:
    fixed: LegConvention
    floating: LegConvention
    spot_lag: int = 2  # business days from trade date to start
    spot_calendar: str | None = None  # None: fixed leg calendar


@dataclass(frozen=True)
class SwapScheduler:
    fixed: LegSchedule
    floating: LegSchedule
    conv: SwapConvention

    @property
    def start(self) -> np.datetime64:
        return self.fixed.adj_start[0]

    @property
    def maturity(self) -> np.datetime64:
        return self.fixed.adj_end[-1]


def make_swap_scheduler(start: dt.date, end: dt.date, conv: SwapConvention, **kwargs) -> SwapScheduler:
    """ Swap between explicit (unadjusted) start and end dates. kwargs go to both legs (see make_leg_schedule) """
    return SwapScheduler(make_leg_schedule(start, end, conv.fixed, **kwargs),
                         make_leg_schedule(start, end, conv.floating, **kwargs), conv)


def make_spot_swap_scheduler(trade_date: dt.date, tenor: str, conv: SwapConvention, forward: str | None = None,
                             **kwargs) -> SwapScheduler:
    """ Market-quoted swap: starts spot_lag business days after trade_date, or forward after spot
        (e.g. forward='5Y', tenor='5Y' for a 5Y5Y), and runs for tenor from its unadjusted start """
    cal = make_calendar(conv.spot_calendar or conv.fixed.calendar)
    start = cal.add_business_days(trade_date, conv.spot_lag)
    if forward is not None:
        start = dts.advance(start, forward)
    end = dts.advance(start, tenor)
    return make_swap_scheduler(start, end, conv, **kwargs)


################## BOND ###########################################################################
@dataclass(frozen=True)
class BondScheduler:
    schedule: LegSchedule
    conv: LegConvention
    anchor: dt.date # date the regular coupon dates are rolled from (ACT/ACT ICMA)

    @property
    def maturity(self) -> np.datetime64:
        """ Redemption payment date, paid together with the last coupon """
        return self.schedule.payment[-1]

    @property
    def coupons_per_year(self) -> int:
        return coupons_per_year(self.conv.frequency)

    def accrual_dates(self) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.datetime64]]:
        """ (start, end) dates the coupons accrue on: unadjusted for bond-style accrual """
        c = self.schedule
        return (c.adj_start, c.adj_end) if self.conv.adjust_accrual else (c.unadj_start, c.unadj_end)

    def accrual_fraction(self, start: dt.date, end: dt.date) -> float:
        """ Accrual fraction between two dates inside the coupon schedule (e.g. last coupon to settlement) """
        return accrual_fractions([start], [end], self.conv, self.anchor)[0]

    def coupon_schedule(self, coupon) -> dict:
        """ Schedule as dataframe """
        c = self.schedule
        cpn_sch = {'Accrual start': c.unadj_start, 'Accrual end': c.unadj_end, 'Payment': c.payment,
                   'Days': (c.unadj_end - c.unadj_start).astype(int), 'Accrual fraction': c.accrual_fraction,
                   'Coupon': 100 * coupon * c.accrual_fraction}
        return cpn_sch


def make_bond_scheduler(issue: dt.date, maturity: dt.date, conv: LegConvention, first_coupon: dt.date | None = None,
                        **kwargs) -> BondScheduler:
    """ Coupon schedule rolled backward from maturity. Irregular first coupon either from issue
        (short_front by default, or stub='long_front') or from an explicit first_coupon date """
    kwargs['first_date'] = first_coupon
    coupons = make_leg_schedule(issue, maturity, conv, **kwargs)
    anchor = to_date(regular_anchor(issue, maturity, **kwargs))
    return BondScheduler(coupons, conv, anchor)


def to_date(d: dt.date | dt.datetime | np.datetime64) -> dt.date:
    """ Plain dt.date from date, datetime or datetime64 """
    if isinstance(d, np.datetime64):
        return d.astype('datetime64[D]').astype(dt.date)
    if isinstance(d, dt.datetime):
        return d.date()
    return d
