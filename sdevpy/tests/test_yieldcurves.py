import datetime as dt
import numpy as np
from sdevpy.market import yieldcurve as ycrv
from sdevpy.tools import timegrids as tg


def test_yieldcurve_creation():
    valdate = dt.datetime(2026, 2, 15)

    # Create curve
    interp_var = 'zerorate'
    scheme = 'linear'
    curve = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=scheme)

    # Create data
    zdates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15), dt.datetime(2056, 2, 15)]
    zrs = np.asarray([0.01, 0.015, 0.020, 0.022, 0.025, 0.030])
    ztimes = tg.model_time(valdate, zdates)
    dfs = np.exp(-zrs * ztimes)

    # Set data
    curve.set_data(zdates, dfs)

    # Interpolate
    test_dates = [dt.datetime(2026, 3, 1), dt.datetime(2026, 6, 15), dt.datetime(2027, 2, 15),
                  dt.datetime(2046, 2, 15)]
    test = curve.discount(test_dates)
    ref = np.asarray([0.99961651, 0.99573301, 0.98019867, 0.57672856])
    assert np.allclose(test, ref, 1e-10)


def test_yieldcurve_reading():
    valdate = dt.datetime(2025, 12, 15)
    name = 'USD.SOFR.1D'

    # Test dates
    zdates = [dt.datetime(2026, 3, 12), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 12), dt.datetime(2056, 2, 27)]
    ztimes = tg.model_time(valdate, zdates)

    # Read curve
    file = ycrv.data_file(name, valdate)
    curve = ycrv.yieldcurve_from_file(file)
    dfs = curve.discount(zdates)
    test = -np.log(dfs) / ztimes
    ref = np.asarray([0.00589787, 0.01375476, 0.01850221, 0.02168115, 0.02991536])
    print(test)
    assert np.allclose(test, ref, 1e-10)
