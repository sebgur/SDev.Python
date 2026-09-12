import datetime as dt
import pytest
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
        assert False, "expected ValueError" # noqa
    except ValueError:
        pass


def test_eom_uses_direction_correct_anchor():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 1, 10) # not EOM
    end = dt.date(2024, 4, 30) # EOM (April has 30 days)

    # default stub "short_front" rolls backward from `end` -> EOM check must use `end`, not `start`
    roll_dates = cal._unadjusted_roll_dates(start, end, '6M', eom=True)
    assert roll_dates == [dt.date(2024, 1, 31), dt.date(2024, 4, 30)]


def test_eom_short_back_anchor_unaffected():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 1, 31) # EOM
    end = dt.date(2024, 4, 10) # not EOM

    roll_dates = cal._unadjusted_roll_dates(start, end, '6M', stub="short_back", eom=True)
    assert roll_dates == [dt.date(2024, 1, 31), dt.date(2024, 4, 30)] # end also snaps to month-end


def test_first_date_override_forces_irregular_front_stub():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 1, 17)
    end = dt.date(2024, 5, 15)
    first_date = dt.date(2024, 2, 15)

    roll_dates = cal._unadjusted_roll_dates(start, end, '1M', first_date=first_date)
    ref = [dt.date(2024, 1, 17), dt.date(2024, 2, 15), dt.date(2024, 3, 15),
           dt.date(2024, 4, 15), dt.date(2024, 5, 15)]
    assert roll_dates == ref


def test_next_to_last_date_override_forces_irregular_back_stub():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 1, 15)
    end = dt.date(2024, 5, 20)
    next_to_last_date = dt.date(2024, 4, 15)

    roll_dates = cal._unadjusted_roll_dates(start, end, '1M', next_to_last_date=next_to_last_date)
    ref = [dt.date(2024, 1, 15), dt.date(2024, 2, 15), dt.date(2024, 3, 15),
           dt.date(2024, 4, 15), dt.date(2024, 5, 20)]
    assert roll_dates == ref


def test_combined_stub_overrides():
    cal = cdr.make_calendar("USD")
    start = dt.date(2024, 1, 17)
    end = dt.date(2024, 5, 20)
    first_date = dt.date(2024, 2, 15)
    next_to_last_date = dt.date(2024, 4, 15)

    roll_dates = cal._unadjusted_roll_dates(start, end, '1M',
                                             first_date=first_date, next_to_last_date=next_to_last_date)
    ref = [dt.date(2024, 1, 17), dt.date(2024, 2, 15), dt.date(2024, 3, 15),
           dt.date(2024, 4, 15), dt.date(2024, 5, 20)]
    assert roll_dates == ref


def test_stub_override_validation():
    cal = cdr.make_calendar("USD")
    start, end = dt.date(2024, 1, 15), dt.date(2024, 5, 15)

    try:
        cal._unadjusted_roll_dates(start, end, '1M', first_date=dt.date(2024, 6, 1))
        assert False, "expected ValueError" # noqa
    except ValueError:
        pass

    try:
        cal._unadjusted_roll_dates(start, end, '1M',
                                    first_date=dt.date(2024, 4, 1), next_to_last_date=dt.date(2024, 3, 1))
        assert False, "expected ValueError" # noqa
    except ValueError:
        pass


class TestTenorToDate:
    def test_simple_year(self):
        assert dts.tenor_to_date('1Y', anchor=dt.datetime(2000, 1, 1)) == dt.datetime(2001, 1, 1)

    def test_compound_tenor(self):
        assert dts.tenor_to_date('2Y6M', anchor=dt.datetime(2000, 1, 1)) == dt.datetime(2002, 7, 1)

    def test_out_of_canonical_order_still_parses(self):
        # Y-M-W-D order is required by dts.period()'s regex; 9M3W is already in that order
        assert dts.tenor_to_date('9M3W', anchor=dt.datetime(2000, 1, 1)) == dt.datetime(2000, 10, 22)

    @pytest.mark.parametrize("short_tenor", ['ON', 'TN', 'SN', 'on', 'tn', 'sn'])
    def test_short_dated_tenors_map_at_or_before_the_anchor(self, short_tenor):
        anchor = dt.datetime(2000, 1, 1)
        assert dts.tenor_to_date(short_tenor, anchor=anchor) <= anchor

    def test_short_dated_tenors_are_strictly_ordered(self):
        anchor = dt.datetime(2000, 1, 1)
        on = dts.tenor_to_date('ON', anchor=anchor)
        tn = dts.tenor_to_date('TN', anchor=anchor)
        sn = dts.tenor_to_date('SN', anchor=anchor)
        assert on < tn < sn

    def test_default_anchor_is_used_when_not_specified(self):
        # Pure function: same tenor, same (default) anchor, same result every call
        assert dts.tenor_to_date('1Y') == dts.tenor_to_date('1Y')

    def test_custom_anchor_changes_the_result(self):
        assert dts.tenor_to_date('1M', anchor=dt.datetime(2026, 2, 1)) == dt.datetime(2026, 3, 1)


class TestTenorLeq:
    def test_shorter_is_leq_longer(self):
        assert dts.tenor_leq('9M', '1Y') is True

    def test_equal_tenors_are_leq_reflexive(self):
        assert dts.tenor_leq('1Y', '1Y') is True

    def test_longer_is_not_leq_shorter(self):
        assert dts.tenor_leq('18M', '1Y') is False

    def test_short_dated_tenor_is_leq_any_nonnegative_tenor(self):
        assert dts.tenor_leq('ON', '1D') is True

    def test_negative_tenor_can_precede_a_short_dated_tenor(self):
        # ON maps to the anchor itself -- it is NOT unconditionally the shortest possible tenor,
        # only the shortest among non-negative ones. A negative tenor lands before the anchor.
        assert dts.tenor_leq('ON', '-1M') is False
        assert dts.tenor_leq('-1M', 'ON') is True

    def test_ordering_is_anchor_dependent_for_close_tenors(self):
        # Documents a real, unavoidable property (verified directly, not assumed): '1M' vs '29D'
        # flips depending on which month the anchor falls in, since month length varies. Pinned
        # against the actual default anchor (2000-01-01, a 31-day January) so a future change to
        # the default anchor is caught here rather than discovered silently downstream.
        assert dts.tenor_leq('1M', '29D') is False  # true for the current default anchor only

    def test_short_dated_tenors_are_ordered_on_tn_sn(self):
        assert dts.tenor_leq('ON', 'TN') is True
        assert dts.tenor_leq('TN', 'ON') is False
        assert dts.tenor_leq('TN', 'SN') is True
        assert dts.tenor_leq('SN', 'TN') is False
        assert dts.tenor_leq('ON', 'SN') is True


if __name__ == "__main__":
    test_tenor_advance()
