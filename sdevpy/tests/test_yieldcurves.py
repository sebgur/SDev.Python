import pytest
import datetime as dt
import numpy as np
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.market import yieldcurve as ycrv
from sdevpy.utilities import timegrids as tg
from sdevpy.market.yieldcurve import InterpolatedYieldCurve
from sdevpy.utilities import timegrids as tg


def _make_curve(interp_var='zerorate', interp_type='linear'):
    valdate = dt.datetime(2026, 1, 1)
    curve = InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=interp_type)
    dates = [dt.datetime(2026, 6, 1), dt.datetime(2027, 1, 1), dt.datetime(2028, 1, 1)]
    dfs = np.asarray([0.99, 0.97, 0.94])
    curve.set_data(dates, dfs)
    return curve, valdate, dates, dfs


def test_yieldcurve_mode():
    curve, valdate, dates, dfs = _make_curve(interp_var='discount')
    # Exact pillar: df should be recovered
    df = curve.discount(dates[0])
    assert abs(df - dfs[0]) < 1e-6

    curve, valdate, dates, dfs = _make_curve(interp_var='log_discount')
    df = curve.discount(dates[1])
    assert abs(df - dfs[1]) < 1e-6

    curve, valdate, dates, dfs = _make_curve(interp_var='zerorate', interp_type='cubicspline')
    df = curve.discount(dates[0])
    assert 0.0 < df < 1.0

    with pytest.raises(RuntimeError):
        InterpolatedYieldCurve(valdate=dt.datetime(2026, 1, 1), interp_var='bogus')

    with pytest.raises(RuntimeError):
        InterpolatedYieldCurve(valdate=dt.datetime(2026, 1, 1),
                               interp_var='zerorate', interp_type='bogus')


def test_yieldcurve_set_data_past_date_raises():
    valdate = dt.datetime(2026, 1, 1)
    curve = InterpolatedYieldCurve(valdate=valdate, interp_var='zerorate')
    with pytest.raises(RuntimeError):
        curve.set_data([dt.datetime(2025, 12, 1)], [0.99])  # date <= valdate


def test_yieldcurve_dump_data_has_pillars():
    curve, valdate, dates, dfs = _make_curve()
    data = curve.dump_data()
    assert 'pillars' in data
    assert len(data['pillars']) == len(dates)


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
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_yieldcurve_reading():
    name, valdate = 'USD.SOFR.1D', dt.datetime(2025, 12, 15)

    # Test dates
    zdates = [dt.datetime(2026, 3, 12), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 12), dt.datetime(2056, 2, 27)]
    ztimes = tg.model_time(valdate, zdates)

    # Read curve
    md = MarketDataFileProvider()
    curve = md.get_yieldcurve(name, valdate)
    dfs = curve.discount(zdates)
    test = -np.log(dfs) / ztimes
    ref = np.asarray([0.00589787, 0.01375476, 0.01850221, 0.02168115, 0.02991536])
    # print(test)
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_yieldcurve_unsorted():
    # Pass same pillars in reverse order; result must be identical
    valdate = dt.datetime(2026, 2, 15)

    # Create curve
    interp_var = 'zerorate'
    scheme = 'linear'
    curve_direct = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=scheme)
    curve_reverse = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var=interp_var, interp_type=scheme)

    # Create data
    zdates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
              dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15), dt.datetime(2056, 2, 15)]
    zrs = np.asarray([0.01, 0.015, 0.020, 0.022, 0.025, 0.030])
    ztimes = tg.model_time(valdate, zdates)
    dfs = np.exp(-zrs * ztimes)

    # Set data
    curve_direct.set_data(zdates, dfs)
    # print(zdates)
    # print(zdates[::-1])
    curve_reverse.set_data(zdates[::-1], dfs[::-1])

    test_dates = [dt.datetime(2026, 3, 1), dt.datetime(2026, 6, 15), dt.datetime(2027, 2, 15),
                  dt.datetime(2046, 2, 15)]

    assert np.allclose(curve_direct.discount(test_dates), curve_reverse.discount(test_dates), rtol=0.0, atol=1e-8)


if __name__ == "__main__":
    test_yieldcurve_unsorted()
