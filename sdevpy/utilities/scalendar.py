import datetime as dt
from datetime import timedelta
from enum import Enum
from collections import namedtuple
import pandas_market_calendars as mcal
import holidays
from sdevpy.utilities import dates as dts
from sdevpy.utilities.tools import isiterable


class BDC(Enum):
    F = "following"
    MF = "modified_following"
    P = "preceding"
    MP = "modified_preceding"
    U = "unadjusted"


Period = namedtuple("Period", ["unadj_start", "unadj_end", "adj_start", "adj_end"])
_IMM_MONTHS = (3, 6, 9, 12)


class Calendar:
    def __init__(self, name, holiday_set: set[dt.date] = None):
        self.name = name
        self._holidays = holiday_set or set()

    def is_holiday(self, d):
        return self.to_date(d) in self._holidays

    def is_weekend(self, d):
        return self.to_date(d).weekday() >= 5

    def is_business_day(self, d):
        return not self.is_weekend(d) and not self.is_holiday(d)

    def adjust(self, d: dt.date, convention: BDC):
        if convention == BDC.U:
            return d
        if self.is_business_day(d):
            return d

        if convention == BDC.F:
            return self._shift(d, 1)
        elif convention == BDC.P:
            return self._shift(d, -1)
        elif convention == BDC.MF:
            adjusted = self._shift(d, 1)
            # Roll back if we've crossed into the next month
            return self._shift(d, -1) if adjusted.month != d.month else adjusted
        elif convention == BDC.MP:
            adjusted = self._shift(d, -1)
            # Roll forward if we've crossed into the previous month
            return self._shift(d, 1) if adjusted.month != d.month else adjusted

    def add_business_days(self, d: dt.date, n: int) -> dt.date:
        step = 1 if n >= 0 else -1
        current = d
        remaining = abs(n)
        while remaining > 0:
            current += timedelta(days=step)
            if self.is_business_day(current):
                remaining -= 1
        return current

    def business_days_between(self, start: dt.date, end: dt.date) -> int:
        step = 1 if end >= start else -1
        current, count = start, 0
        while current != end:
            current += timedelta(days=step)
            if self.is_business_day(current):
                count += step
        return count

    # def make_schedule(self, start: dt.date, end: dt.date, term: str, convention: BDC = BDC.MF,
    #                         stub: str = "short_front", # short_front|long_front|short_back|long_back
    #                         eom: bool = False, convert_to_datetime: bool = False) -> list[dt.date]:
    #     """ Generate a schedule of adjusted dates from start to end """
    #     period = dts.period(term)
    #     use_eom = eom and is_eom(start)

    #     # 1) Generate unadjusted roll dates
    #     if stub in ("short_back", "long_back"):
    #         # Roll forward from start
    #         roll_dates, d = [start], start
    #         while True:
    #             d += period
    #             roll_dates.append(min(d, end))
    #             if d >= end:
    #                 break
    #         if stub == "long_back" and len(roll_dates) > 2:
    #             roll_dates.pop(-2) # merge last two periods into one long stub
    #     else:
    #         # Roll backwards from end (industry standard for front stubs)
    #         roll_dates, d = [end], end
    #         while True:
    #             d -= period
    #             roll_dates.insert(0, max(d, start))
    #             if d <= start:
    #                 break
    #         if stub == "long_front" and len(roll_dates) > 2:
    #             roll_dates.pop(1) # merge first two periods into one long stub

    #     # 2) Apply end-of-month roll
    #     if use_eom:
    #         roll_dates = [to_eom(d) for d in roll_dates]

    #     # 3) Adjust all dates in one pass
    #     period_ends = roll_dates[:] # Effectively take them all
    #     seen = set()
    #     adjusted = []
    #     for d in period_ends:
    #         a = self.adjust(d, convention)
    #         if a not in seen:
    #             adjusted.append(a)
    #             seen.add(a)

    #     return to_datetime(adjusted) if convert_to_datetime else adjusted

    def _unadjusted_roll_dates(self, start: dt.date, end: dt.date, term: str, stub: str="short_front",
                               eom: bool=False, roll_convention: str=None) -> list[dt.date]:
        """ Generate deduplicated unadjusted roll dates from start to end (no BDC applied) """
        period = dts.period(term)
        use_eom = eom and is_eom(start)

        if roll_convention == "IMM":
            total_months = period.years * 12 + period.months
            if total_months <= 0 or total_months % 3 != 0 or period.days != 0:
                raise ValueError(f"IMM roll convention requires a term in whole 3-month steps, got '{term}'")
            n_quarters = total_months // 3
            def step(d):
                return _step_imm(d, n_quarters, forward=True)

            def step_back(d):
                return _step_imm(d, n_quarters, forward=False)
        else:
            def step(d):
                return d + period

            def step_back(d):
                return d - period

        if stub in ("short_back", "long_back"):
            roll_dates, d = [start], start
            while True:
                d = step(d)
                # d += period
                roll_dates.append(min(d, end))
                if d >= end:
                    break
            if stub == "long_back" and len(roll_dates) > 2:
                roll_dates.pop(-2)
        else:
            roll_dates, d = [end], end
            while True:
                d = step_back(d)
                # d -= period
                roll_dates.insert(0, max(d, start))
                if d <= start:
                    break
            if stub == "long_front" and len(roll_dates) > 2:
                roll_dates.pop(1)

        if use_eom:
            roll_dates = [to_eom(d) for d in roll_dates]

        seen = set()
        deduped = []
        for d in roll_dates:
            if d not in seen:
                deduped.append(d)
                seen.add(d)
        return deduped

    def make_schedule(self, start: dt.date, end: dt.date, term: str, convention: BDC=BDC.MF,
                      stub: str="short_front", eom: bool=False, convert_to_datetime: bool=False,
                      termination_convention: BDC=None, roll_convention: str=None) -> list[dt.date]:
        """ Generate a schedule of adjusted dates from start to end """
        termination_convention = termination_convention or convention
        roll_dates = self._unadjusted_roll_dates(start, end, term, stub, eom, roll_convention)

        seen = set()
        adjusted = []
        for i, d in enumerate(roll_dates):
            conv = termination_convention if i == len(roll_dates) - 1 else convention
            a = self.adjust(d, conv)
            if a not in seen:
                adjusted.append(a)
                seen.add(a)

        return to_datetime(adjusted) if convert_to_datetime else adjusted

    def make_periods(self, start: dt.date, end: dt.date, term: str, convention: BDC = BDC.MF,
                     stub: str="short_front", eom: bool=False, termination_convention: BDC=None,
                     roll_convention: str=None) -> list["Period"]:
        """ Generate (unadjusted, adjusted) date pairs for each accrual period """
        termination_convention = termination_convention or convention
        roll_dates = self._unadjusted_roll_dates(start, end, term, stub, eom, roll_convention)

        # adjusted = [self.adjust(d, convention) for d in roll_dates]
        adjusted = [self.adjust(d, convention) for d in roll_dates[:-1]]
        adjusted.append(self.adjust(roll_dates[-1], termination_convention))

        return [Period(roll_dates[i], roll_dates[i + 1], adjusted[i], adjusted[i + 1])
                for i in range(len(roll_dates) - 1)]

    def __add__(self, other: "Calendar") -> "Calendar":
        """ Combine two calendars — both holidays are observed """
        return Calendar(name=f"{self.name}+{other.name}", holiday_set=self._holidays | other._holidays)

    def _shift(self, d: dt.date, step: int):
        current = d + timedelta(days=step)
        while not self.is_business_day(current):
            current += timedelta(days=step)
        return current

    def __repr__(self):
        return f"Calendar('{self.name}', {len(self._holidays)} holidays)"

    def to_date(self, d: dt.date | dt.datetime):
        if isinstance(d, dt.datetime):
            return d.date()
        else:
            return d


def is_eom(d: dt.date) -> bool:
    """ True if d is the last calendar day of its month """
    return (d + timedelta(days=1)).month != d.month


def to_eom(d: dt.date) -> dt.date:
    """ Roll d to the last calendar day of its month """
    next_month = d.replace(day=28) + timedelta(days=4)
    return next_month - timedelta(days=next_month.day)


def third_wednesday(year: int, month: int) -> dt.date:
    """ Third Wednesday of the given month (standard IMM roll day) """
    d = dt.date(year, month, 1)
    wednesdays = [d + timedelta(days=i) for i in range(31)
                if (d + timedelta(days=i)).month == month
                and (d + timedelta(days=i)).weekday() == 2]
    return wednesdays[2]


def next_imm_date(d: dt.date) -> dt.date:
    """ First IMM date strictly after d """
    y, m = d.year, d.month
    for im in _IMM_MONTHS:
        if im > m or (im == m and third_wednesday(y, im) > d):
            return third_wednesday(y, im)
    return third_wednesday(y + 1, _IMM_MONTHS[0])


def prev_imm_date(d: dt.date) -> dt.date:
    """ First IMM date strictly before d """
    y, m = d.year, d.month
    for im in reversed(_IMM_MONTHS):
        if im < m or (im == m and third_wednesday(y, im) < d):
            return third_wednesday(y, im)
    return third_wednesday(y - 1, _IMM_MONTHS[-1])


def _step_imm(d: dt.date, n: int, forward: bool) -> dt.date:
    for _ in range(n):
        d = next_imm_date(d) if forward else prev_imm_date(d)
    return d


def make_calendar(name: str, start_year: int=2000, end_year: int=2100):
    if ',' in name:
        cdr_names = name.split(',')
        cdrs = [make_calendar(cdrn, start_year=start_year, end_year=end_year) for cdrn in cdr_names]
        cdr = cdrs[0]
        for i in range(1, len(cdrs)):
            cdr = cdr + cdrs[i]
        return cdr

    # Single calendar name
    y = range(start_year, end_year + 1)

    if name not in CCY_CALENDARS:
        return make_calendar_from_mcal(name, y)
    else:
        hols = set()
        for year in y:
            hols.update(CCY_CALENDARS[name](year).keys())

        return Calendar(name, hols)


def make_calendar_from_mcal(exchange, years: range):
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=f"{years.start}-01-01", end_date=f"{years.stop - 1}-12-31")

    # Get all valid trading days, then infer holidays as non-trading weekdays
    trading_days = set(schedule.index.date)
    all_weekdays = {
        d for y in years
        for d in (dt.date(y, 1, 1) + timedelta(days=i) for i in range(366))
        if d.year == y and d.weekday() < 5
    }

    holidays = all_weekdays - trading_days
    return Calendar(name=exchange, holiday_set=holidays)


def list_mcal_calendars() -> list[str]:
    """ List all calendars available in mcal """
    return mcal.get_calendar_names()


def make_schedule(calstr, start, end, term, convention=BDC.F, convert_to_datetime=False):
    """ Helper wrapping around the make_schedule() method of the calendard class """
    cdr = make_calendar(calstr)
    schedule = cdr.make_schedule(start, end, term, convention=convention,
                                 convert_to_datetime=convert_to_datetime)
    return schedule


def to_datetime(date) -> dt.datetime:
    """ Convert date to datetime """
    if isiterable(date):
        datetimes = [dt.datetime.combine(d, dt.time.min) for d in date]
        return datetimes
    else:
        return dt.datetime.combine(date, dt.time.min)


CCY_CALENDARS = {
        "ARS": lambda y: holidays.AR(years=y),
        "AUD": lambda y: holidays.AU(years=y),
        "BRL": lambda y: holidays.BR(years=y),
        "CAD": lambda y: holidays.CA(years=y),
        "CHF": lambda y: holidays.CH(years=y),
        "CLP": lambda y: holidays.CL(years=y),
        "CNY": lambda y: holidays.CN(years=y),
        "COP": lambda y: holidays.CO(years=y),
        "CZK": lambda y: holidays.CZ(years=y),
        "DKK": lambda y: holidays.DK(years=y),
        "EGP": lambda y: holidays.EG(years=y),
        "EUR": lambda y: holidays.ECB(years=y), # TARGET calendar
        "GBP": lambda y: holidays.UK(years=y),
        "HKD": lambda y: holidays.HK(years=y),
        "HUF": lambda y: holidays.HU(years=y),
        "IDR": lambda y: holidays.ID(years=y),
        "ILS": lambda y: holidays.IL(years=y),
        "INR": lambda y: holidays.IN(years=y),
        "JPY": lambda y: holidays.JP(years=y),
        "KRW": lambda y: holidays.KR(years=y),
        "KZT": lambda y: holidays.KZ(years=y),
        "MXN": lambda y: holidays.MX(years=y),
        "MYR": lambda y: holidays.MY(years=y),
        "NOK": lambda y: holidays.NO(years=y),
        "NZD": lambda y: holidays.NZ(years=y),
        "PHP": lambda y: holidays.PH(years=y),
        "PLN": lambda y: holidays.PL(years=y),
        "RUB": lambda y: holidays.RU(years=y),
        "SAR": lambda y: holidays.SA(years=y),
        "SEK": lambda y: holidays.SE(years=y),
        "SGD": lambda y: holidays.SG(years=y),
        "THB": lambda y: holidays.TH(years=y),
        "TRY": lambda y: holidays.TR(years=y),
        "TWD": lambda y: holidays.TW(years=y),
        "USD": lambda y: holidays.US(years=y),
        "VND": lambda y: holidays.VN(years=y),
        "WE": lambda y: {},
        "ZAR": lambda y: holidays.ZA(years=y),
    }

if __name__ == "__main__":
    # View holidays
    # loader = lambda y: holidays.US(years=y)
    # hols = set()
    # hols.update(loader(2024).keys())
    # print(hols)

    # Usage
    usd_cal = make_calendar("USD,EUR")
    gbp_cal = make_calendar("GBP")
    cal = usd_cal + gbp_cal
    cal = make_calendar("WE")

    # Adjust a date
    raw_date = dt.date(2024, 12, 25)
    print(cal.is_holiday(raw_date))
    print(cal.is_holiday(dt.datetime(2024, 12, 25)))
    adjusted = cal.adjust(raw_date, BDC.MF)
    print(adjusted)

    start = dt.date(2024, 1, 15)
    end = dt.date(2025, 1, 15)
    schedule = cal.make_schedule(start, end, '3M', convention=BDC.MF)
    print(schedule)

    # Usage
    nyse_cal = make_calendar_from_mcal("NYSE", range(2020, 2030))
    lse_cal  = make_calendar_from_mcal("LSE", range(2020, 2030))

    joint_cal = nyse_cal + lse_cal  # composability still works

    # Eom
    d = dt.datetime(2026, 2, 27)
    print(is_eom(d))
    print(to_eom(d))

    # Schedule
    start = dt.datetime(2025, 11, 15)
    end = dt.datetime(2026, 12, 15)
    sch1 = make_schedule("USD", start, end, "1d")
    # sch2 = cal.make_schedule(start, end, "1d")
    print(sch1)
    # print(sch2)

