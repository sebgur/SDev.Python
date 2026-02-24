import datetime as dt
from sdevpy.tools import dates


def test_date_advance():
    base = dt.datetime(2026, 2, 15)
    test = dates.date_advance(base, years=-1)
    ref = dt.datetime(2025, 2, 15)
    assert test == ref


if __name__ == "__main__":
    test_date_advance()
