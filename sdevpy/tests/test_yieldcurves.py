import datetime as dt
import numpy as np
from sdevpy.market import yieldcurve as ycrv
from sdevpy.tools import timegrids as tg


def test_yieldcurve_creation():
    valdate = dt.datetime(2026, 2, 15)

    # Create curve
    interp_var = 'zerorate'
    scheme = 'linear'
    curve = ycrv.create_yieldcurve(valdate, interp_var, scheme)

    # Create data
    zdates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15), dt.datetime(2056, 2, 15)]
    zrs = np.asarray([0.01, 0.015, 0.020, 0.022, 0.025, 0.030])
    ztimes = tg.model_time(valdate, zdates)
    dfs = np.exp(-zrs * ztimes)

    # Set data
    curve.set_data(zdates, dfs)

    # Interpolate and display
    test_dates = [dt.datetime(2026, 3, 1), dt.datetime(2026, 6, 15), dt.datetime(2027, 2, 15),
                  dt.datetime(2046, 2, 15)]
    test = curve.discount(test_dates)
    ref = np.asarray([0.99961651, 0.99573301, 0.98019867, 0.57672856])
    assert np.allclose(test, ref, 1e-10)
