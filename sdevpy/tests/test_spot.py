import datetime as dt
from sdevpy.market.spot import get_spots


def test_spotdata():
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)

    # Fetch data
    test = get_spots([name], valdate)[0]
    ref = 100.0
    assert test == ref
