import numpy as np
import pandas as pd
import pytest
from sdevpy.timeseries.meanreversion import (
    MeanRevertingTimeSeries,
    compute_mean_reversion_params,
    mr_expected_and_variance_change,
    compute_sharpe_ratio,
)


MR_LEVEL = 100.0
MR_RATE = -0.008  # negative → mean-reverting
NORMAL_VOL = 2.0


######### HELPERS #################################################################################
def _make_ou_series(n=252 * 5, s0=100.0, sbar=100.0, kappa=2.0,
                    sigma_daily=2.0, seed=42) -> pd.Series:
    """ Discrete OU process: ds = kappa*(sbar-s)*dt + sigma*dW, dt=1/252. """
    rng = np.random.RandomState(seed)
    dt = 1.0 / 252.0
    vals = [s0]
    for g in rng.normal(size=n):
        ds = kappa * (sbar - vals[-1]) * dt + sigma_daily * g
        vals.append(vals[-1] + ds)
    return pd.Series(vals)


######### TESTS #################################################################################
def test_mr_compute_mr_params_rate_is_negative():
    s = _make_ou_series()
    res = compute_mean_reversion_params(s)
    assert res['MR Rate'] < 0.0


def test_mr_compute_params_half_life_is_positive():
    s = _make_ou_series()
    res = compute_mean_reversion_params(s)
    assert res['Half Life'] > 0.0


def test_mr_compute_params_half_life_formula():
    s = _make_ou_series()
    res = compute_mean_reversion_params(s)
    assert np.isclose(res['Half Life'], -np.log(2) / res['MR Rate'])


# def test_mr_compute_params_level_close_to_true_mean():
#     # With 5 years of daily data the OLS estimate should be within 2 units of true mean 100
#     s = _make_ou_series(sbar=100.0)
#     res = compute_mean_reversion_params(s)
#     assert abs(res['MR Level'] - 100.0) < 2.0


def test_mr_compute_params_pvalues_in_unit_interval():
    s = _make_ou_series()
    res = compute_mean_reversion_params(s)
    assert 0.0 < res['Const p-value'] < 1.0
    assert 0.0 < res['Series p-value'] < 1.0


def test_mr_ev_at_time_zero():
    edx, vardx = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 0, 95.0, NORMAL_VOL)
    assert np.isclose(edx, 0.0)
    assert np.isclose(vardx, 0.0)


def test_mr_ev_below_mean_expects_positive_return():
    edx, _ = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL)
    assert edx > 0.0


def test_mr_ev_above_mean_expects_negative_return():
    edx, _ = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 5, 110.0, NORMAL_VOL)
    assert edx < 0.0


def test_mr_ev_at_mean_expects_zero_return():
    edx, _ = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 5, MR_LEVEL, NORMAL_VOL)
    assert np.isclose(edx, 0.0)


def test_mr_ev_variance_increases_with_time():
    _, var5 = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL)
    _, var10 = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 10, 90.0, NORMAL_VOL)
    assert var10 > var5


def test_mr_sharpe_return_keys():
    res = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL, current_zscore=-1.0)
    assert set(res.keys()) == {'Sharpe Ratio', 'Return Expectation', 'Return SD'}


def test_mr_sharpe_return_sd_positive():
    res = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL, current_zscore=-1.0)
    assert res['Return SD'] > 0.0


def test_mr_sharpe_positive_when_buying_below_mean():
    # below mean → zscore < 0 → we buy → price expected to rise → positive sharpe
    res = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL, current_zscore=-1.0)
    assert res['Sharpe Ratio'] > 0.0
    assert res['Return Expectation'] > 0.0


def test_mr_sharpe_positive_when_shorting_above_mean():
    # above mean → zscore > 0 → we short → price expected to fall → positive sharpe
    res = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 110.0, NORMAL_VOL, current_zscore=+1.0)
    assert res['Sharpe Ratio'] > 0.0
    assert res['Return Expectation'] > 0.0


def test_mr_sharpe_expectation_flips_sign_with_zscore():
    buy = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL, current_zscore=-1.0)
    sell = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL, current_zscore=+1.0)
    assert np.isclose(buy['Return Expectation'], -sell['Return Expectation'])


def test_mr_sharpe_sd_consistent_with_variance():
    _, vardx = mr_expected_and_variance_change(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL)
    res = compute_sharpe_ratio(MR_LEVEL, MR_RATE, 5, 90.0, NORMAL_VOL, current_zscore=-1.0)
    assert np.isclose(res['Return SD'], np.sqrt(vardx))


@pytest.fixture(scope='module')
def mr_ts():
    return MeanRevertingTimeSeries(_make_ou_series(sbar=100.0))


def test_mr_ts_rate_negative(mr_ts):
    assert mr_ts.get_mr_rate() < 0.0


# def test_mr_ts_level_close_to_true_mean(mr_ts):
#     assert abs(mr_ts.get_mr_level() - 100.0) < 2.0


def test_mr_ts_current_level_is_last_value(mr_ts):
    s = _make_ou_series(sbar=100.0)
    obj = MeanRevertingTimeSeries(s)
    assert np.isclose(obj.get_current_level(), s.iloc[-1])


def test_mr_ts_get_level_at_t(mr_ts):
    s = _make_ou_series(sbar=100.0)
    obj = MeanRevertingTimeSeries(s)
    idx = s.index[10]
    assert np.isclose(obj.get_level_at_t(idx), s.iloc[10])


def test_mr_ts_current_zscore_formula(mr_ts):
    s = _make_ou_series(sbar=100.0)
    obj = MeanRevertingTimeSeries(s)
    expected = (s.iloc[-1] - obj.get_mr_level()) / obj.get_stdev()
    assert np.isclose(obj.get_current_zscore(), expected)


def test_mr_ts_pvalues_in_unit_interval(mr_ts):
    assert 0.0 < mr_ts.get_const_pvalue() < 1.0
    assert 0.0 < mr_ts.get_series_pvalue() < 1.0
