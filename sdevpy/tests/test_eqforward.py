import datetime as dt
import numpy as np
from sdevpy.market import yieldcurve as ycrv
from sdevpy.market import eqforward as eqf


def test_eqforward_creation():
    name = "ABC"
    valdate = dt.datetime(2026, 2, 15)
    spot = 100.0
    file = eqf.data_file(name, valdate)

    # Get data from existing file
    test_data = eqf.eqforwarddata_from_file(file)

    # Create forward curve
    curve = eqf.EqForwardCurve(valdate=valdate, interp_var='forward', interp_type='cubicspline')
    yieldcurve = ycrv.get_yieldcurve('USD.SOFR.1D', valdate)
    curve.calibrate(test_data, spot, yieldcurve)

    # Interpolate and display
    test_dates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
                  dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15)]

    test = curve.value(test_dates)
    ref = np.asarray([100.07674176, 100.74660895, 102.020134, 111.6345355, 128.42013226])
    assert np.allclose(test, ref, 1e-10)
