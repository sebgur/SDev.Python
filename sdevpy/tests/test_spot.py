import datetime as dt
from sdevpy.market import spot as spt



def test_spotdata():
    name = "ABC"
    valdate = dt.datetime(2026, 2, 15)

    # Get data from existing file
    file = spt.data_file(name, valdate)
    test_data = spt.spotdata_from_file(file)
    test = test_data.value
    ref = 100.0
    assert test == ref
