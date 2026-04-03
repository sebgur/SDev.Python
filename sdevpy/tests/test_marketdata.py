import datetime as dt
import numpy as np
from sdevpy.market.spot import get_spots
from sdevpy.market import yieldcurve as ycrv
from sdevpy.market import eqforward as eqf
from sdevpy.market import correlations


def test_correlations():
    names = ['ABC', 'KLM', 'XYZ']
    valdate = dt.datetime(2025, 12, 15)
    c = correlations.get_correlations(names, valdate)
    ref = np.asarray([0.5, 0.1, 0.1])
    test = np.asarray([c[0, 1], c[0, 2], c[1, 2]])
    assert np.allclose(test, ref, 1e-10)


def test_spotdata():
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)

    # Fetch data
    test = get_spots([name], valdate)[0]
    ref = 100.0
    assert test == ref


def test_eqforward_creation():
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)
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
    ref = np.asarray([100.28716535, 101.09321315, 102.21072051, 111.87417472, 128.4201])
    assert np.allclose(test, ref, 1e-10)
