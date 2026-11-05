import datetime as dt
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
import numpy as np
import numpy.typing as npt
from sdevpy.utilities.scalendar import BDC, Period, make_calendar, is_eom, to_eom
from sdevpy.utilities.sdaycount import DayCount, yearfraction, icma_yearfraction, act365_canadian
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
class YieldType(Enum):
    STREET = "street"  # compounded at the coupon frequency per coupon period
    SIMPLE = "simple"  # (c + (R - clean) / T) / clean: JGB simple yield; money-market yield (R/P - 1) / T for bills


@dataclass(frozen=True)
class BondConvention:
    """ Bond market rules on top of the coupon leg """
    leg: LegConvention  # coupon schedule and accrual
    settlement_lag: int = 1  # business days from trade date, on leg.calendar
    ex_coupon_days: int = 0  # business days before a coupon date from which a buyer no longer gets it (0: none)
    ex_coupon_calendar_days: bool = False  # ex_coupon_days in calendar days before the coupon date (ACGB)
    simple_final_period: bool = False  # simple-interest yield in the last coupon period
    final_period_daycount: DayCount | None = None  # simple_final_period on this day count (None: coupon time)
    accrued_daycount: DayCount | None = None  # accrued interest day count when it differs from the coupon's
    accrued_cap: bool = False  # accrued interest capped at the current period's coupon (JGB: c d / 365, at most c/2)
    stub_daycount: DayCount | None = None  # irregular coupon amounts: 1/f per full quasi-period, broken part on this
    full_first_coupon: bool = False  # first coupon a full regular one, accruing from the coupon date before issue
    yield_type: YieldType = YieldType.STREET  # the market's quoted yield
    yield_frequency: int | None = None  # street-yield compounding per year (None: coupon frequency; BTP: 1)

    @property
    def compounding(self) -> int:
        """ Street-yield compounding frequency per year """
        return self.yield_frequency or coupons_per_year(self.leg.frequency)


@dataclass(frozen=True)
class BillConvention:
    """ Zero-coupon bills: the market's quoted yield """
    yield_type: YieldType = YieldType.STREET  # STREET: US bond-equivalent; SIMPLE: money-market, P = R / (1 + y T)
    daycount: DayCount | None = None  # SIMPLE: T on this day count (None: bonds.simple_yield_time, JGB)


@dataclass(frozen=True)
class BondScheduler:
    schedule: LegSchedule
    conv: BondConvention
    anchor: dt.date # date the regular coupon dates are rolled from (ACT/ACT ICMA)

    @property
    def maturity(self) -> np.datetime64:
        """ Redemption payment date, paid together with the last coupon """
        return self.schedule.payment[-1]

    @property
    def coupons_per_year(self) -> int:
        return coupons_per_year(self.conv.leg.frequency)

    @cached_property
    def coupon_fraction(self) -> npt.NDArray[np.float64]:
        """ Coupon amount per unit coupon rate, per period: the leg's accrual fractions, except for the irregular
            first/last periods under conv.stub_daycount """
        if self.conv.stub_daycount is None:
            return self.schedule.accrual_fraction
        starts, ends = self.accrual_dates()
        alpha = self.schedule.accrual_fraction.copy()
        for i in {0, len(alpha) - 1}:
            alpha[i] = stub_coupon_fraction(to_date(starts[i]), to_date(ends[i]), self.anchor, self.conv)
        return alpha

    def accrual_dates(self) -> tuple[npt.NDArray[np.datetime64], npt.NDArray[np.datetime64]]:
        """ (start, end) dates the coupons accrue on: unadjusted for bond-style accrual """
        c = self.schedule
        return (c.adj_start, c.adj_end) if self.conv.leg.adjust_accrual else (c.unadj_start, c.unadj_end)

    def ex_coupon_date(self, k: int) -> dt.date | None:
        """ First settlement date on which a buyer no longer receives coupon k: conv.ex_coupon_days business days
            before its payment date (gilts), or calendar days before its coupon date (ACGB: 7, i.e. settlement after
            the record date, 8 calendar days before rolled to the preceding business day). None: no ex-coupon period """
        if self.conv.ex_coupon_days == 0:
            return None
        if self.conv.ex_coupon_calendar_days:
            return to_date(self.schedule.unadj_end[k]) - dt.timedelta(days=self.conv.ex_coupon_days)
        cal = make_calendar(self.conv.leg.calendar)
        return cal.add_business_days(to_date(self.schedule.payment[k]), -self.conv.ex_coupon_days)

    def accrual_fraction(self, start: dt.date, end: dt.date) -> float:
        """ Accrual fraction between two dates inside the coupon schedule (e.g. last coupon to settlement) """
        return accrual_fractions([start], [end], self.conv.leg, self.anchor)[0]

    def accrued_fraction(self, start: dt.date, end: dt.date) -> float:
        """ Year fraction for accrued interest from start (last coupon date) to end (settlement): the leg's
            accrual unless the convention has a separate accrued day count (e.g. JGB NL/365) """
        dc = self.conv.accrued_daycount
        if dc is None:
            return self.accrual_fraction(start, end)
        if dc == DayCount.ACT365_CANADIAN:
            leg = self.conv.leg
            quasi = quasi_coupon_dates(self.anchor, start, end, leg.frequency, leg.eom)
            return act365_canadian(start, end, quasi, coupons_per_year(leg.frequency))
        return float(yearfraction(start, end, dc))

    def coupon_schedule(self, coupon) -> dict:
        """ Schedule as dataframe """
        c = self.schedule
        cpn_sch = {'Accrual start': c.unadj_start, 'Accrual end': c.unadj_end, 'Payment': c.payment,
                   'Days': (c.unadj_end - c.unadj_start).astype(int), 'Accrual fraction': c.accrual_fraction,
                   'Coupon': 100 * coupon * c.coupon_fraction}
        return cpn_sch


def make_bond_scheduler(issue: dt.date, maturity: dt.date, conv: BondConvention, first_coupon: dt.date | None = None,
                        **kwargs) -> BondScheduler:
    """ Coupon schedule rolled backward from maturity. Irregular first coupon either from issue
        (short_front by default, or stub='long_front') or from an explicit first_coupon date """
    if conv.full_first_coupon:  # accrual from the regular coupon date on or before issue (JGB)
        issue = quasi_coupon_dates(to_date(maturity), to_date(issue), to_date(issue), conv.leg.frequency,
                                   conv.leg.eom)[0]

    kwargs['first_date'] = first_coupon
    coupons = make_leg_schedule(issue, maturity, conv.leg, **kwargs)
    anchor = to_date(regular_anchor(issue, maturity, **kwargs))
    return BondScheduler(coupons, conv, anchor)


def stub_coupon_fraction(start: dt.date, end: dt.date, anchor: dt.date, conv: BondConvention) -> float:
    """ Coupon per unit rate of a possibly irregular period: 1/f for each full quasi-coupon period inside
        [start, end], conv.stub_daycount for the broken parts (e.g. Canada: Act/365 Canadian bond per quasi-period) """
    leg = conv.leg
    f = coupons_per_year(leg.frequency)
    quasi = quasi_coupon_dates(anchor, start, end, leg.frequency, leg.eom)

    if conv.stub_daycount == DayCount.ACT365_CANADIAN:
        return act365_canadian(start, end, quasi, f)

    total = 0.0
    for q0, q1 in zip(quasi[:-1], quasi[1:], strict=True):
        s, e = max(start, q0), min(end, q1)
        if (s, e) == (q0, q1):
            total += 1.0 / f
        elif e > s:
            total += float(yearfraction(s, e, conv.stub_daycount))
    return total


def to_date(d: dt.date | dt.datetime | np.datetime64) -> dt.date:
    """ Plain dt.date from date, datetime or datetime64 """
    if isinstance(d, np.datetime64):
        return d.astype('datetime64[D]').astype(dt.date)
    if isinstance(d, dt.datetime):
        return d.date()
    return d
