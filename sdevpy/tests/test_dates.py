import datetime as dt
from sdevpy.utilities import dates as dts
from sdevpy.utilities import scalendar as cdr


def test_to_oadate():
    d = dt.datetime(2026, 3, 8)
    test = dts.to_oadate(d)
    ref = 46089.0
    assert abs(test - ref) < 1e-12


def test_tenor_advance():
    base = dt.datetime(2025, 12, 15)
    tenors = ['-1D', '1D', '2W', '1M', '2Y', '1Y6M']
    test = [base + dts.period(t) for t in tenors]
    # print(test)
    ref = [dt.datetime(2025, 12, 14), dt.datetime(2025, 12, 16), dt.datetime(2025, 12, 29),
           dt.datetime(2026, 1, 15), dt.datetime(2027, 12, 15), dt.datetime(2027, 6, 15)]
    assert test == ref


def test_make_schedule():
    cal1 = cdr.make_calendar("USD")
    cal2 = cdr.make_calendar("NYSE")
    cal = cal1 + cal2

    start = dt.date(2024, 1, 15)
    end = dt.date(2025, 1, 15)
    test = cal.make_schedule(start, end, '3M')
    # print(test)
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
    test = dts.advance(base, '-1y')
    ref = dt.datetime(2025, 2, 15)
    assert test == ref


def test_make_periods():
    cal1 = cdr.make_calendar("USD")
    cal2 = cdr.make_calendar("NYSE")
    cal = cal1 + cal2

    start = dt.date(2024, 1, 15)
    end = dt.date(2025, 1, 15)
    periods = cal.make_periods(start, end, '3M')

    # 2024-01-15 is MLK Day (holiday) -> adjusted start rolls to the 16th under MF
    assert periods[0].unadj_start == dt.date(2024, 1, 15)
    assert periods[0].adj_start == dt.date(2024, 1, 16)

    # every other roll date in this run is already a business day -> unadjusted == adjusted
    for p in periods[1:]:
        assert p.unadj_start == p.adj_start

    # adjacent periods share the same roll date and therefore the same adjustment
    for i in range(len(periods) - 1):
        assert periods[i].adj_end == periods[i + 1].adj_start

    # flat adjusted schedule from make_schedule must match the period boundaries
    flat = cal.make_schedule(start, end, '3M')
    rebuilt = [periods[0].adj_start] + [p.adj_end for p in periods]
    assert flat == rebuilt


def test_make_schedule_termination_convention():
    cal1 = cdr.make_calendar("USD")
    cal2 = cdr.make_calendar("NYSE")
    cal = cal1 + cal2

    start = dt.date(2024, 1, 15)
    end = dt.date(2025, 1, 18) # Saturday

    default_sched = cal.make_schedule(start, end, '3M')
    assert default_sched[-1] == dt.date(2025, 1, 21) # MF rolls Sat -> Mon

    unadj_end_sched = cal.make_schedule(start, end, '3M', termination_convention=cdr.BDC.U)
    assert unadj_end_sched[-1] == end  # maturity stays exactly as booked
    assert unadj_end_sched[:-1] == default_sched[:-1]  # interior dates unaffected

    periods = cal.make_periods(start, end, '3M', termination_convention=cdr.BDC.U)
    assert periods[-1].adj_end == end
    assert periods[-1].unadj_end == end


def test_third_wednesday():
    assert cdr.third_wednesday(2024, 3) == dt.date(2024, 3, 20)
    assert cdr.third_wednesday(2024, 6) == dt.date(2024, 6, 19)
    assert cdr.third_wednesday(2024, 9) == dt.date(2024, 9, 18)
    assert cdr.third_wednesday(2024, 12) == dt.date(2024, 12, 18)


def test_next_imm_date():
    assert cdr.next_imm_date(dt.date(2024, 3, 1)) == dt.date(2024, 3, 20)
    assert cdr.next_imm_date(dt.date(2024, 3, 20)) == dt.date(2024, 6, 19)  # on an IMM date -> rolls to next
    assert cdr.next_imm_date(dt.date(2024, 12, 19)) == dt.date(2025, 3, 19)  # wraps into next year


def test_prev_imm_date():
    assert cdr.prev_imm_date(dt.date(2024, 3, 25)) == dt.date(2024, 3, 20)
    assert cdr.prev_imm_date(dt.date(2024, 3, 20)) == dt.date(2023, 12, 20)  # on an IMM date -> rolls to prior
    assert cdr.prev_imm_date(dt.date(2024, 1, 1)) == dt.date(2023, 12, 20)


def test_make_periods_imm_roll_backward():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 3, 20)   # already an IMM date
    end = dt.date(2025, 3, 19)     # IMM date one year later (2025 IMM day differs: 19th not 20th)

    roll_dates = cal._unadjusted_roll_dates(start, end, '3M', stub="short_front", roll_convention="IMM")
    ref = [dt.date(2024, 3, 20), dt.date(2024, 6, 19), dt.date(2024, 9, 18),
           dt.date(2024, 12, 18), dt.date(2025, 3, 19)]
    assert roll_dates == ref


def test_make_periods_imm_roll_forward():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 3, 20)
    end = dt.date(2025, 3, 19)

    roll_dates = cal._unadjusted_roll_dates(start, end, '3M', stub="short_back", roll_convention="IMM")
    ref = [dt.date(2024, 3, 20), dt.date(2024, 6, 19), dt.date(2024, 9, 18),
           dt.date(2024, 12, 18), dt.date(2025, 3, 19)]
    assert roll_dates == ref


def test_make_periods_imm_rejects_non_quarterly_term():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 3, 20)
    end = dt.date(2025, 3, 19)

    try:
        cal._unadjusted_roll_dates(start, end, '1M', roll_convention="IMM")
        assert False, "expected ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_tenor_advance()
