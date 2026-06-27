import pytest
import datetime as dt
import numpy as np
from sdevpy.market import provider as mdp
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.market import eqforward as eqf
from sdevpy.market.fixings import FixingHandler, data_file
# from sdevpy.tests.conftest import marketdata_path


def _make_handler(interpolate=False):
    dates = [dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 2), dt.datetime(2025, 12, 3)]
    values = [100.0, 101.0, 102.0]
    return FixingHandler("TEST", dates, values, interpolate=interpolate)


def test_fixinghandler_scalar_lookup():
    h = _make_handler()
    assert h.value(dt.datetime(2025, 12, 2)) == 101.0


def test_fixinghandler_list_lookup():
    h = _make_handler()
    result = h.value([dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 3)])
    assert result == [100.0, 102.0]


def test_fixinghandler_missing_raises():
    h = _make_handler(interpolate=False)
    with pytest.raises(ValueError):
        h.value(dt.datetime(2025, 12, 5))


def test_fixinghandler_unsorted_input_stores_sorted():
    dates = [dt.datetime(2025, 12, 3), dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 2)]
    values = [102.0, 100.0, 101.0]
    h = FixingHandler("TEST", dates, values)
    assert h.dates[0] == dt.datetime(2025, 12, 1)
    assert h.values[0] == 100.0


def test_fixinghandler_interpolation():
    h = _make_handler(interpolate=True)
    # dt.datetime(2025, 12, 1) and (2025, 12, 3) are exact; interpolated mid should be ~101
    result = h.value(dt.datetime(2025, 12, 2))
    assert abs(result - 101.0) < 0.5


def test_data_file_returns_correct_path():
    from pathlib import Path
    p = data_file("ABC", folder=Path("/some/folder"))
    assert p == Path("/some/folder/ABC.csv")


def test_correlations():
    names = ['ABC', 'KLM', 'XYZ']
    valdate = dt.datetime(2025, 12, 15)
    md = MarketDataFileProvider()
    c = md.get_correlations(names, valdate)
    # c = correlations.get_correlations(names, valdate)
    ref = np.asarray([0.5, 0.1, 0.1])
    test = np.asarray([c[0, 1], c[0, 2], c[1, 2]])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_spotdata():
    name, valdate = "ABC", dt.datetime(2025, 12, 15)

    # Fetch data
    md = MarketDataFileProvider()
    test = md.get_spot(name, valdate)
    ref = 100.0
    assert test == ref


def test_eqforward_creation():
    name, valdate = "ABC", dt.datetime(2025, 12, 15)
    spot = 100.0

    # Get data from existing file
    md = MarketDataFileProvider()
    test_data = md.get_eq_forward_data(name, valdate)

    # Create forward curve
    curve = eqf.EqForwardCurve(valdate=valdate, interp_var='forward', interp_type='cubicspline')
    yieldcurve = md.get_yieldcurve('USD.SOFR.1D', valdate)
    curve.calibrate(test_data, spot, yieldcurve)

    # Interpolate and display
    test_dates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
                  dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15)]

    test = curve.value(test_dates)
    ref = np.asarray([100.50233117, 101.4947209, 102.06711626, 112.73343447, 128.4201])
    # ref = np.asarray([100.28716535, 101.09321315, 102.21072051, 111.87417472, 128.4201])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_eq_option_strikes():
    name, valdate = "ABC", dt.datetime(2025, 12, 15)

    # Retrieve market option data object
    md = MarketDataFileProvider()
    vol_data = md.get_eq_vol_data(name, valdate)

    # Retrieve forward curve
    fwd_curve = mdp.get_eq_forward_curves([name], valdate, md)[0]

    # Access data in object
    test = vol_data.get_strikes(fwd_curve, 'absolute')
    # print(test)

    ref = np.asarray([[90.26318122, 94.70076604, 99.88756326, 105.35844335, 110.53815253],
                      [84.95857122, 91.65636925, 99.71914514, 108.49118276, 117.04419888],
                      [79.28085982, 88.26212825, 99.43907907, 112.03140738, 124.72279524],
                      [71.77471782, 83.53767673, 98.88130446, 117.04314454, 136.22502001],
                      [62.15139633, 77.03054626, 97.77512372, 124.10628358, 153.81753883],
                      [46.17835038, 64.83651535, 94.53027807, 137.82316066, 193.51001927]])
    assert(np.allclose(test, ref, rtol=0.0, atol=1e-8))

    test = vol_data.get_strikes(fwd_curve, 'relative')
    # print(test)

    ref = np.asarray([[0.90082835, 0.94511554, 0.99687988, 1.05147937, 1.10317297],
                      [0.84534839, 0.91199231, 0.99221794, 1.07950081, 1.16460439],
                      [0.78492002, 0.87383905, 0.98449644, 1.10916676, 1.23481783],
                      [0.70353483, 0.81883520, 0.96923323, 1.14725535, 1.33527584],
                      [0.59714405, 0.74010135, 0.93941306, 1.19240007, 1.47786267],
                      [0.41783899, 0.58666505, 0.85534533, 1.24707553, 1.75095106]])
    assert(np.allclose(test, ref, rtol=0.0, atol=1e-8))


if __name__ == "__main__":
    test_eq_option_strikes()
