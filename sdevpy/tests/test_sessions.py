import datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd
import pytest
from sdevpy.utilities import sessions as ssn
from sdevpy.utilities import calendars as cdr


NY = ZoneInfo("America/New_York")
TOKYO = ZoneInfo("Asia/Tokyo")
THANKSGIVING = dt.date(2026, 11, 26)
FRI_1500 = dt.datetime(2026, 11, 13, 15, 0, tzinfo=NY)
SAT_1200 = dt.datetime(2026, 11, 14, 12, 0, tzinfo=NY)
SUN_1730 = dt.datetime(2026, 11, 15, 17, 30, tzinfo=NY)


def test_normalize_date():
    d = dt.date(2026, 11, 9)
    test = ssn.normalize(d)
    assert test == d
    assert type(test) is dt.date


def test_normalize_naive_midnight():
    test = ssn.normalize(dt.datetime(2026, 11, 9))
    assert test == dt.date(2026, 11, 9)
    assert type(test) is dt.date


@pytest.mark.parametrize('t', [dt.datetime(2026, 11, 9, 16, 30), dt.datetime(2026, 11, 9, 0, 0, 1),
                               dt.datetime(2026, 11, 9, 0, 0, 0, 1)])
def test_normalize_naive_non_midnight_raises(t):
    with pytest.raises(ValueError):
        ssn.normalize(t)


@pytest.mark.parametrize('t', [dt.datetime(2026, 11, 9, 16, 30, tzinfo=NY),
                               dt.datetime(2026, 11, 9, tzinfo=NY),
                               dt.datetime(2026, 11, 9, tzinfo=dt.UTC)])
def test_normalize_aware_is_instant(t):
    test = ssn.normalize(t)
    assert test is t


def test_normalize_pandas_timestamp():
    assert ssn.normalize(pd.Timestamp('2026-11-09')) == dt.date(2026, 11, 9)
    t = pd.Timestamp('2026-11-09 16:30', tz='America/New_York')
    assert ssn.normalize(t) is t


@pytest.mark.parametrize('x', ['2026-11-09', 46335.0, None])
def test_normalize_bad_type_raises(x):
    with pytest.raises(TypeError):
        ssn.normalize(x)


def _ny_center():
    return ssn.TradingCenter("NY", NY, cdr.Calendar("TEST", {THANKSGIVING}))


@pytest.mark.parametrize('t, ref', [
    (dt.datetime(2026, 11, 13, 10, 0), dt.date(2026, 11, 13)),       # Fri, in session
    (dt.datetime(2026, 11, 13, 16, 59, 59), dt.date(2026, 11, 13)),  # Fri, just before roll
    (dt.datetime(2026, 11, 13, 17, 0), dt.date(2026, 11, 13)),       # Fri roll: Sat, back to Fri
    (dt.datetime(2026, 11, 14, 12, 0), dt.date(2026, 11, 13)),       # Sat
    (dt.datetime(2026, 11, 15, 16, 59), dt.date(2026, 11, 13)),      # Sun, before roll
    (dt.datetime(2026, 11, 15, 17, 0), dt.date(2026, 11, 16)),       # Sun roll: Mon session starts
    (dt.datetime(2026, 11, 16, 18, 0), dt.date(2026, 11, 17)),       # Mon evening: Tue
    (dt.datetime(2026, 11, 25, 18, 0), dt.date(2026, 11, 25)),       # Wed evening: Thu holiday, back to Wed
    (dt.datetime(2026, 11, 26, 12, 0), dt.date(2026, 11, 25)),       # Thanksgiving
    (dt.datetime(2026, 11, 26, 17, 0), dt.date(2026, 11, 27)),       # Thanksgiving roll: Fri
])
def test_session_date_ny(t, ref):
    center = _ny_center()
    assert center.session_date(t.replace(tzinfo=NY)) == ref


def test_session_date_dst():
    # Same UTC time of day, either side of the US DST change (1 Nov 2026)
    center = _ny_center()
    before = dt.datetime(2026, 10, 26, 21, 30, tzinfo=dt.UTC)  # 17:30 EDT, after roll
    after = dt.datetime(2026, 11, 2, 21, 30, tzinfo=dt.UTC)    # 16:30 EST, before roll
    assert center.session_date(before) == dt.date(2026, 10, 27)
    assert center.session_date(after) == dt.date(2026, 11, 2)


def test_session_date_other_zone():
    # Tokyo Tue 08:00 = NY Mon 18:00 (EST): NY session is Tue
    center = _ny_center()
    t = dt.datetime(2026, 11, 17, 8, 0, tzinfo=TOKYO)
    assert center.session_date(t) == dt.date(2026, 11, 17)


def test_session_date_naive_raises():
    center = _ny_center()
    with pytest.raises(ValueError):
        center.session_date(dt.datetime(2026, 11, 13, 10, 0))


@pytest.mark.parametrize('x, now, ref', [
    (dt.date(2026, 11, 13), FRI_1500, True),                         # Fri date, Fri session
    (dt.datetime(2026, 11, 13), FRI_1500, True),                     # naive midnight = date
    (dt.date(2026, 11, 14), FRI_1500, True),                         # Sat date, back to Fri
    (dt.date(2026, 11, 12), FRI_1500, False),                        # Thu date
    (dt.date(2026, 11, 16), FRI_1500, False),                        # Mon date
    (dt.datetime(2026, 11, 13, 10, 0, tzinfo=NY), FRI_1500, True),   # Fri morning
    (dt.datetime(2026, 11, 12, 17, 30, tzinfo=NY), FRI_1500, True),  # Thu after roll: Fri session
    (dt.datetime(2026, 11, 12, 16, 0, tzinfo=NY), FRI_1500, False),  # Thu before roll
    (dt.date(2026, 11, 13), SAT_1200, True),                         # weekend: Fri still live
    (dt.date(2026, 11, 16), SAT_1200, False),
    (dt.date(2026, 11, 16), SUN_1730, True),                         # Sun after roll: Mon live
    (dt.date(2026, 11, 13), SUN_1730, False),
])
def test_is_live(x, now, ref):
    center = _ny_center()
    assert center.is_live(x, now) == ref


def test_is_live_system_clock():
    center = _ny_center()
    assert center.is_live(dt.datetime.now(dt.UTC))


def test_is_live_naive_now_raises():
    center = _ny_center()
    with pytest.raises(ValueError):
        center.is_live(dt.date(2026, 11, 13), now=dt.datetime(2026, 11, 13, 15, 0))
