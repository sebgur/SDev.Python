import datetime as dt
from sdevpy.tools import dates
from sdevpy.tools import scalendar as cdr


def test_make_schedule():
    cal1 = cdr.make_calendar("USD")
    cal2 = cdr.make_calendar("NYSE")
    cal = cal1 + cal2

    start = dt.date(2024, 1, 15)
    end = dt.date(2025, 1, 15)
    test = cdr.make_schedule(start, end, freq_months=3, cal=cal)
    ref = [dt.date(2024, 1, 16), dt.date(2024, 4, 15), dt.date(2024, 7, 15),
           dt.date(2024, 10, 15), dt.date(2025, 1, 15)]
    assert test == ref


def test_calendar_adjust():
    usd_cal = cdr.make_calendar("USD")
    gbp_cal = cdr.make_calendar("GBP")
    cal = usd_cal + gbp_cal

    # Adjust a date
    raw_date = dt.date(2024, 12, 25)
    test = cal.adjust(raw_date, cdr.BDC.MF)
    ref = dt.date(2024, 12, 27)
    assert test == ref


def test_date_advance():
    base = dt.datetime(2026, 2, 15)
    test = dates.date_advance(base, years=-1)
    ref = dt.datetime(2025, 2, 15)
    assert test == ref


if __name__ == "__main__":
    test_date_advance()
