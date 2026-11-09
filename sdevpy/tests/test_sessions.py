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
CHICAGO = ZoneInfo("America/Chicago")
TOKYO_WED_1000 = dt.datetime(2026, 11, 11, 10, 0, tzinfo=TOKYO)
TOKYO_SAT_1000 = dt.datetime(2026, 11, 14, 10, 0, tzinfo=TOKYO)

@pytest.fixture
def naive_tokyo():
    ssn.set_naive_tz("Asia/Tokyo")
    yield
    ssn.set_naive_tz(None)


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
def test_normalize_naive_non_midnight(t, naive_tokyo):
    test = ssn.normalize(t)
    assert test == t.replace(tzinfo=TOKYO)
    assert test.utcoffset() is not None


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


def test_session_date_naive(naive_tokyo):
    center = _ny_center()
    assert center.session_date(dt.datetime(2026, 11, 13, 6, 0)) == dt.date(2026, 11, 12)


@pytest.mark.parametrize('x, now, ref', [
    (dt.date(2026, 11, 13), FRI_1500, True),                         # Fri date, Fri session
    (dt.datetime(2026, 11, 13), FRI_1500, True),                     # naive midnight = date
    (dt.date(2026, 11, 14), FRI_1500, False),                        # Sat date: gap, never live
    (dt.datetime(2026, 11, 14, 12, 0, tzinfo=NY), FRI_1500, False),  # Sat moment: gap, never live
    (dt.date(2026, 11, 12), FRI_1500, False),                        # Thu date
    (dt.date(2026, 11, 16), FRI_1500, False),                        # Mon date
    (dt.datetime(2026, 11, 13, 10, 0, tzinfo=NY), FRI_1500, True),   # Fri morning
    (dt.datetime(2026, 11, 12, 17, 30, tzinfo=NY), FRI_1500, True),  # Thu after roll: Fri session
    (dt.datetime(2026, 11, 12, 16, 0, tzinfo=NY), FRI_1500, False),  # Thu before roll
    (dt.date(2026, 11, 13), SAT_1200, False),                        # weekend gap: nothing live
    (dt.date(2026, 11, 16), SAT_1200, False),
    (dt.date(2026, 11, 16), SUN_1730, True),                         # Sun after roll: Mon live
    (dt.date(2026, 11, 13), SUN_1730, False),
])
def test_is_live(x, now, ref):
    center = _ny_center()
    assert center.is_live(x, now) == ref


def test_is_live_system_clock():
    center = _ny_center()
    now = dt.datetime.now(dt.UTC)
    assert center.is_live(now) == center.in_session(now)


def test_is_live_naive_now(naive_tokyo):
    center = _ny_center()
    assert center.is_live(dt.date(2026, 11, 13), now=dt.datetime(2026, 11, 14, 5, 0))


@pytest.mark.parametrize('name', list(ssn.CENTER_SPECS))
def test_make_center(name):
    center = ssn.make_center(name)
    assert center.name == name
    assert center is ssn.make_center(name)  # cached


def test_make_center_unknown_raises():
    with pytest.raises(ValueError):
        ssn.make_center("XXXX")


@pytest.mark.parametrize('name, t, ref', [
    ("XNYS", dt.datetime(2026, 11, 25, 18, 0, tzinfo=NY), dt.date(2026, 11, 25)),     # Thanksgiving: back to Wed
    ("XTKS", dt.datetime(2026, 11, 22, 17, 0, tzinfo=TOKYO), dt.date(2026, 11, 20)),  # Mon 23 holiday: back to Fri
    ("XTKS", dt.datetime(2026, 11, 23, 17, 0, tzinfo=TOKYO), dt.date(2026, 11, 24)),  # holiday roll: Tue
    ("CME", dt.datetime(2026, 11, 16, 17, 30, tzinfo=CHICAGO), dt.date(2026, 11, 17)),  # Chicago roll
    ("CME", dt.datetime(2026, 11, 16, 17, 30, tzinfo=NY), dt.date(2026, 11, 16)),     # 16:30 Chicago
    ("FX", dt.datetime(2026, 11, 14, 12, 0, tzinfo=NY), dt.date(2026, 11, 13)),       # Sat: back to Fri
    ("FX", dt.datetime(2026, 11, 15, 17, 0, tzinfo=NY), dt.date(2026, 11, 16)),       # Sun roll: Mon
])
def test_registry_session_date(name, t, ref):
    assert ssn.make_center(name).session_date(t) == ref


@pytest.fixture
def no_default():
    ssn.set_default_center(None)
    yield
    ssn.set_default_center(None)


def test_set_default_center(no_default):
    ssn.set_default_center("LONDON")
    assert ssn.default_center().name == "LONDON"
    ssn.set_default_center(None)
    assert ssn.default_center().name == "NEWYORK"


def test_set_default_center_unknown_raises(no_default):
    with pytest.raises(ValueError):
        ssn.set_default_center("PARIS")


def test_is_live_branches(no_default):
    # Fri 13 Nov 2026 10:00 UTC = 19:00 Tokyo (after EOD), 10:00 London (before EOD)
    now = dt.datetime(2026, 11, 13, 10, 0, tzinfo=dt.UTC)
    fri = dt.date(2026, 11, 13)
    ssn.set_default_center("TOKYO")
    assert not ssn.is_live(fri, now=now)                   # Tokyo: after Fri EOD, weekend gap
    assert ssn.is_live(fri, center="LONDON", now=now)
    thu_now = dt.datetime(2026, 11, 12, 10, 0, tzinfo=dt.UTC)  # Thu 19:00 Tokyo, 10:00 London
    thu = dt.date(2026, 11, 12)
    assert not ssn.is_live(thu, now=thu_now)               # Tokyo already on Fri
    assert ssn.is_live(thu, center="LONDON", now=thu_now)  # London still on Thu


@pytest.mark.parametrize('end_date, now, ref', [
    (None, TOKYO_WED_1000, (dt.date(2026, 11, 11), True)),
    ('live', TOKYO_WED_1000, (dt.date(2026, 11, 11), True)),
    ('now', TOKYO_WED_1000, (dt.date(2026, 11, 11), True)),
    (dt.datetime(2026, 11, 11), TOKYO_WED_1000, (dt.date(2026, 11, 11), True)),   # naive midnight = date
    (dt.datetime(2026, 11, 10), TOKYO_WED_1000, (dt.date(2026, 11, 10), False)),  # past date
    (None, TOKYO_SAT_1000, (dt.date(2026, 11, 13), False)),                       # weekend: Fri, not live
    (dt.date(2026, 11, 14), TOKYO_SAT_1000, (dt.date(2026, 11, 13), False)),      # Sat date: Fri, not live
    (dt.datetime(2026, 11, 11, 18, 0, tzinfo=TOKYO), TOKYO_WED_1000, (dt.date(2026, 11, 12), False)),  # after EOD
])
def test_resolve_date_input(end_date, now, ref):
    assert ssn.resolve_date_input(end_date, center="TOKYO", now=now) == ref


@pytest.mark.parametrize('end_date', ['today', 'yesterday'])
def test_resolve_date_input_invalid_raises(end_date):
    with pytest.raises(ValueError):
        ssn.resolve_date_input(end_date, center="TOKYO", now=TOKYO_WED_1000)


@pytest.mark.parametrize('t, ref', [
    (dt.datetime(2026, 11, 13, 16, 59), True),   # Fri before EOD
    (dt.datetime(2026, 11, 13, 17, 0), False),   # Fri EOD: weekend gap starts
    (dt.datetime(2026, 11, 14, 12, 0), False),   # Sat
    (dt.datetime(2026, 11, 15, 16, 59), False),  # Sun before roll
    (dt.datetime(2026, 11, 15, 17, 0), True),    # Sun roll: Mon session starts
    (dt.datetime(2026, 11, 16, 18, 0), True),    # Mon evening: Tue session
    (dt.datetime(2026, 11, 25, 18, 0), False),   # Wed evening: Thanksgiving gap
    (dt.datetime(2026, 11, 26, 17, 0), True),    # Thanksgiving roll: Fri session
])
def test_in_session(t, ref):
    center = _ny_center()
    assert center.in_session(t.replace(tzinfo=NY)) == ref


def test_in_session_naive(naive_tokyo):
    center = _ny_center()
    assert center.in_session(dt.datetime(2026, 11, 14, 6, 0))
