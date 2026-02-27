import datetime as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from enum import Enum
from functools import lru_cache
import pandas_market_calendars as mcal
import holidays
from sdevpy.tools.utils import isiterable
from sdevpy.tools import dates, speriods


class BDC(Enum):
    F = "following"
    MF = "modified_following"
    P = "preceding"
    MP = "modified_preceding"
    U = "unadjusted"


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

    def add_business_days(self, d: dt.date, n: int) -> date:
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

    def __add__(self, other: "Calendar") -> "Calendar":
        """Combine two calendars — both holidays are observed."""
        return Calendar(name=f"{self.name}+{other.name}", holiday_set=self._holidays | other._holidays,)

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


def make_calendar(name, start_year=2000, end_year=2100):
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


def list_mcal_calendars():
    print(mcal.get_calendar_names())


# ToDo: this looks like a very simplified function. Make more precise one.
def make_schedule(cal, start, end, term, convention=BDC.F, convert_to_datetime=False):
    schedule_dates, d = [], start
    while d <= end:
        schedule_dates.append(cal.adjust(d, convention))
        d += speriods.period(term)
        # d = dates.date_advance(d, days=freq_days, months=freq_months, years=freq_years)#relativedelta(months=freq_months)
        # # d += relativedelta(months=freq_months)

    return to_datetime(schedule_dates) if convert_to_datetime else schedule_dates


def list_dates(start_date, end_date, cdr=None):
    """ List dates between start and end, including both if they are business days. """
    dates = []
    date = start_date if is_business_day(start_date, cdr) else next_business_day(start_date, cdr)
    while date <= end_date:
        dates.append(date)
        date = next_business_day(date, cdr)

    return dates

def to_datetime(date):
    if isiterable(date):
        datetimes = [dt.datetime.combine(d, dt.time.min) for d in date]
        return datetimes
    else:
        return dt.datetime.combine(d, time.min)


CCY_CALENDARS = {
        "WE": lambda y: {},
        "USD": lambda y: holidays.US(years=y),
        "GBP": lambda y: holidays.UK(years=y),
        "EUR": lambda y: holidays.ECB(years=y),  # TARGET calendar
        "JPY": lambda y: holidays.JP(years=y),
    }


if __name__ == "__main__":
    # View holidays
    # loader = lambda y: holidays.US(years=y)
    # hols = set()
    # hols.update(loader(2024).keys())
    # print(hols)

    # Usage
    usd_cal = make_calendar("USD")
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
    schedule = make_schedule(cal, start, end, freq_months=3, convention=BDC.MF)
    print(schedule)

    # Usage
    nyse_cal = make_calendar_from_mcal("NYSE", range(2020, 2030))
    lse_cal  = make_calendar_from_mcal("LSE", range(2020, 2030))

    joint_cal = nyse_cal + lse_cal  # composability still works

    # list_mcal_calendars()
