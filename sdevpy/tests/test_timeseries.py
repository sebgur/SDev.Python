import numpy as np
import pandas as pd
from sdevpy.timeseries.timeseriestools import (
    compute_diff, min_max_return_x_days, min_max_return_x_days_in_sd,
    x_day_historical_returns, x_day_historical_returns_in_sd,
    is_inverted_quote, weighted_series, last_daily_hist_normal_vol, daily_hist_normal_vol
)


def _series(n=20, start=100.0, step=1.0):
    idx = pd.date_range('2024-01-01', periods=n, freq='B')
    return pd.Series(start + step * np.arange(n), index=idx, name='test')


def test_compute_diff_no_shift_shape():
    s = _series()
    df = compute_diff(s, num_of_days=3, to_shift=0)  # ToShiftOrNot.NOT_SHIFT = 0
    assert df.shape[1] == 3


def test_compute_diff_with_shift_shape():
    from sdevpy.timeseries.timeseriestools import ToShiftOrNot
    s = _series()
    df = compute_diff(s, num_of_days=2, to_shift=ToShiftOrNot.TO_SHIFT)
    assert df.shape[1] == 2


def test_compute_diff_linear_series():
    # Linear series: diff(1) should be constant = step
    from sdevpy.timeseries.timeseriestools import ToShiftOrNot
    s = _series(step=2.0)
    df = compute_diff(s, num_of_days=1, to_shift=ToShiftOrNot.NOT_SHIFT)
    vals = df.iloc[:, 0].dropna()
    assert np.allclose(vals, 2.0)


def test_min_max_return_x_days_columns():
    s = _series()
    df = min_max_return_x_days(s, time_in_days=2)
    assert 'max' in df.columns and 'min' in df.columns
    assert (df['max'] >= df['min']).all()


def test_min_max_return_x_days_in_sd():
    s = _series()
    res = min_max_return_x_days_in_sd(s, time_in_days=2, basket_stdev=2.0)
    ref = min_max_return_x_days(s, time_in_days=2) / 2.0
    assert np.allclose(res['max'].values, ref['max'].values)


def test_x_day_historical_returns_shape():
    s = _series(n=30)
    res = x_day_historical_returns(s, time_in_days=5)
    assert len(res) == len(s)

    s = _series(n=30)
    raw = x_day_historical_returns(s, time_in_days=3)
    scaled = x_day_historical_returns_in_sd(s, time=3, basket_stdev=5.0)
    assert np.allclose(raw.dropna().values, (scaled.dropna() * 5.0).values)


def test_is_inverted_quote():
    assert not is_inverted_quote('EURUSD Curncy')
    assert is_inverted_quote('JPYUSD Curncy')
    assert is_inverted_quote('XYZUSD Curncy') is None


def test_weighted_series_shape():
    idx = pd.date_range('2024-01-01', periods=10, freq='B')
    df = pd.DataFrame({'A': np.ones(10), 'B': 2 * np.ones(10)}, index=idx)
    s = weighted_series(df, [1.0, 0.5])
    assert len(s) == 10
    assert np.allclose(s.values, 2.0)  # 1*1 + 0.5*2


def test_last_daily_hist_normal_vol_positive():
    s = _series(n=30, step=0.5)
    vol = last_daily_hist_normal_vol(s, period=15)
    assert vol >= 0.0
    vol = daily_hist_normal_vol(s, period=5)
    assert len(vol) == len(s)
