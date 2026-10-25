import pytest
import datetime as dt
import numpy as np
from sdevpy.utilities import timegrids as tg
from sdevpy.pricingcontext import default_calibration_repository
from sdevpy.calibration.rates import yieldcurve as ycrv
from sdevpy.calibration.rates.yieldcurve import InterpolatedYieldCurve
from sdevpy.calibration.rates.curvebasis import CurveBasis


VALDATE = dt.datetime(2025, 12, 15)
DATES = [VALDATE + dt.timedelta(days=round(365 * x)) for x in [0.5, 1, 2, 5, 10, 20, 30]]
TIMES = tg.model_time(VALDATE, DATES)
DFS = np.exp(-(0.036 + 0.012 * (1 - np.exp(-TIMES / 4))) * TIMES)
PER_END = ('not-a-knot', 'natural')


def make_curve(**kwargs):
    curve = InterpolatedYieldCurve(valdate=VALDATE, **kwargs)
    curve.set_data(DATES, DFS)
    return curve


def log_df_second_derivative(curve, t, e=1e-4):
    return (np.log(curve.discount_float(t + e)) - 2 * np.log(curve.discount_float(t))
            + np.log(curve.discount_float(t - e))) / e ** 2


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

    curve, valdate, dates, dfs = _make_curve(interp_var='logdiscount')
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
    calib = default_calibration_repository()[valdate]
    curve = calib.get_yieldcurve(name)
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


def test_per_end_conditions_applied():
    """ Natural end: log-DF'' = 0 at the last node (flat forward); not-a-knot start: log-DF'' free at t = 0 """
    per_end = make_curve(interp_var='logdiscount', interp_type='cubicspline-linear', bc_type=PER_END)
    natural = make_curve(interp_var='logdiscount', interp_type='cubicspline-linear')
    assert abs(log_df_second_derivative(per_end, TIMES[-1] - 1e-3)) < 1e-5
    assert abs(log_df_second_derivative(natural, 1e-4)) < 1e-5
    assert abs(log_df_second_derivative(per_end, 1e-4)) > 1e-4
    np.testing.assert_allclose(per_end.discount_float(TIMES), DFS, rtol=0, atol=1e-14)  # still through the nodes


def test_default_unchanged():
    """ No bc_type: zero-rate cubic spline with flat extrapolation stays clamped (z' = 0 at the ends) """
    curve = make_curve(interp_var='zerorate', interp_type='cubicspline')
    assert curve.bc_type is None
    z = lambda t: -np.log(curve.discount_float(t)) / t
    assert abs((z(TIMES[-1]) - z(TIMES[-1] - 1e-4)) / 1e-4) < 1e-6


def test_bc_type_only_for_cubic_spline():
    with pytest.raises(ValueError):
        InterpolatedYieldCurve(valdate=VALDATE, interp_var='logdiscount', interp_type='pchip-linear', bc_type=PER_END)


@pytest.mark.parametrize("bc_type", [None, PER_END, ((1, 0.0), 'natural')])
def test_dump_and_reload(tmp_path, bc_type):
    curve = make_curve(interp_var='logdiscount', interp_type='cubicspline-linear', bc_type=bc_type)
    file = tmp_path / 'curve.json'
    curve.dump(str(file))
    loaded = ycrv.yieldcurve_from_file(file)
    assert loaded.bc_type == curve.bc_type
    t = np.linspace(0.01, 35.0, 200)
    np.testing.assert_allclose(loaded.discount_float(t), curve.discount_float(t), rtol=0, atol=1e-15)
    assert ('bc_type' in curve.dump_data()) == (bc_type is not None)  # files without bc_type unchanged


def test_basis_stays_linear():
    curve = InterpolatedYieldCurve(valdate=VALDATE, interp_var='logdiscount', interp_type='cubicspline-linear',
                                   bc_type=PER_END)
    assert CurveBasis(curve, TIMES, np.linspace(0.0, 30.0, 50)).linear


if __name__ == "__main__":
    test_yieldcurve_unsorted()
