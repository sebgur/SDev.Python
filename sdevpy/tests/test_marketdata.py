import datetime as dt
import numpy as np
from sdevpy.market.spot import get_spots
from sdevpy.market import yieldcurve as ycrv
from sdevpy.market import eqforward as eqf
from sdevpy.market import eqvolsurface as vsurf
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
    ref = np.asarray([100.50233117, 101.4947209, 102.06711626, 112.73343447, 128.4201])
    # ref = np.asarray([100.28716535, 101.09321315, 102.21072051, 111.87417472, 128.4201])
    assert np.allclose(test, ref, 1e-10)


def test_eq_option_strikes():
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)

    # Retrieve market option data object
    file = vsurf.data_file(name, valdate)
    vol_data = vsurf.eqvolsurfacedata_from_file(file)

    # Retrieve forward curve
    fwd_curve = eqf.get_forward_curves([name], valdate)[0]

    # Access data in object
    # expiries = vol_data.expiries
    test = vol_data.get_strikes('absolute')
    # print(test)
    test = vol_data.get_strikes2(fwd_curve, 'absolute')
    # print(ref2)
    # assert(np.allclose(test, ref2, 1e-10))

    # print(test)
    ref = np.asarray([[90.26318122, 94.70076604, 99.88756326, 105.35844335, 110.53815253],
                      [84.95857122, 91.65636925, 99.71914514, 108.49118276, 117.04419888],
                      [79.28085982, 88.26212825, 99.43907907, 112.03140738, 124.72279524],
                      [71.77471782, 83.53767673, 98.88130446, 117.04314454, 136.22502001],
                      [62.15139633, 77.03054626, 97.77512372, 124.10628358, 153.81753883],
                      [46.17835038, 64.83651535, 94.53027807, 137.82316066, 193.51001927]])
    assert(np.allclose(test, ref, 1e-10))

    # test = vol_data.get_strikes('relative')
    # print(test)
    # print('ref2')
    test = vol_data.get_strikes2(fwd_curve, 'relative')
    # print(ref2)
    # assert(np.allclose(test, ref2, 1e-10))

    ref = np.asarray([[0.90082835, 0.94511554, 0.99687988, 1.05147937, 1.10317297],
                      [0.84534839, 0.91199231, 0.99221794, 1.07950081, 1.16460439],
                      [0.78492002, 0.87383905, 0.98449644, 1.10916676, 1.23481783],
                      [0.70353483, 0.81883520, 0.96923323, 1.14725535, 1.33527584],
                      [0.59714405, 0.74010135, 0.93941306, 1.19240007, 1.47786267],
                      [0.41783899, 0.58666505, 0.85534533, 1.24707553, 1.75095106]])
    assert(np.allclose(test, ref, 1e-10))


if __name__ == "__main__":
    test_eq_option_strikes()
