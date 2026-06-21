import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sdevpy.timeseries.cointegration import johansen_test, check_johansen_stats_fast
from sdevpy.timeseries.cointegration import (
    norm_1st_eigvec, trace_stats, eigen_stats, coint_diagnostics
)


# Fixed seed — strong cointegration so results are deterministic
# _RNG = np.random.default_rng(42)


def make_cointegrated(n=1000, a=2.0, noise_scale=0.005, seed=42):
    """Two cointegrated series: y2 = a * y1 + small_noise.
    Cointegrating vector (normalised): [1, -1/a].
    """
    rng = np.random.default_rng(seed)
    # if rng is None:
    #     rng = _RNG
    y1 = np.cumsum(rng.normal(0, 1, n))
    y2 = a * y1 + rng.normal(0, noise_scale, n)
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame({"y1": y1, "y2": y2}, index=dates)


_MOD = "sdevpy.timeseries.cointegration"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def diag_inputs():
    """Run Johansen once and share across all coint_diagnostics tests."""
    df = make_cointegrated(n=300)
    res_jo = coint_johansen(df, 0, 1)
    weights = norm_1st_eigvec(res_jo)
    test = {
        'trace (5%)': True, 'trace (10%)': True,
        'eigen (5%)': True, 'eigen (10%)': True,
        'weights': weights, '_raw': res_jo,
    }
    return df, test


# ── norm_1st_eigvec ───────────────────────────────────────────────────────────

def _mock_evec(evec_array):
    m = MagicMock()
    m.evec = np.array(evec_array, dtype=float)
    return m


class TestNorm1stEigvec:
    def test_first_element_is_one(self):
        result = norm_1st_eigvec(_mock_evec([[2.0, 0.5], [3.0, 1.0]]))
        assert result[0] == pytest.approx(1.0)

    def test_second_element_is_ratio(self):
        # evec[0][0]=2.0, evec[1][0]=3.0 → 3.0/2.0 = 1.5
        result = norm_1st_eigvec(_mock_evec([[2.0, 0.5], [3.0, 1.0]]))
        assert result[1] == pytest.approx(1.5)

    def test_three_assets(self):
        # evec[:,0] = [4, 2, 1] → normalised = [1, 0.5, 0.25]
        result = norm_1st_eigvec(_mock_evec([[4.0, 0.1, 0.2],
                                             [2.0, 0.3, 0.4],
                                             [1.0, 0.5, 0.6]]))
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(0.25)

    def test_returns_numpy_array(self):
        result = norm_1st_eigvec(_mock_evec([[1.0, 0.0], [0.5, 0.0]]))
        assert isinstance(result, np.ndarray)


# ── trace_stats ───────────────────────────────────────────────────────────────

def _mock_trace(lr1, cvt):
    m = MagicMock()
    m.lr1 = lr1
    m.cvt = cvt
    return m


class TestTraceStats:
    _lr1 = [50.0, 10.0]
    _cvt = [[13.4, 15.5, 19.9], [2.7, 3.8, 6.6]]

    def test_columns(self):
        df = trace_stats(_mock_trace(self._lr1, self._cvt))
        assert list(df.columns) == ['trace', '10%', '5%', '1%']

    def test_index_labels(self):
        df = trace_stats(_mock_trace(self._lr1, self._cvt))
        assert list(df.index) == ['r=0', 'r<=1']

    def test_shape(self):
        df = trace_stats(_mock_trace(self._lr1, self._cvt))
        assert df.shape == (2, 4)

    def test_trace_stats_populated(self):
        df = trace_stats(_mock_trace(self._lr1, self._cvt))
        assert df.loc['r=0',  'trace'] == pytest.approx(50.0)
        assert df.loc['r<=1', 'trace'] == pytest.approx(10.0)

    def test_critical_values_populated(self):
        df = trace_stats(_mock_trace([50.0], [[13.4, 15.5, 19.9]]))
        assert df.loc['r=0', '10%'] == pytest.approx(13.4)
        assert df.loc['r=0',  '5%'] == pytest.approx(15.5)
        assert df.loc['r=0',  '1%'] == pytest.approx(19.9)

    def test_three_assets_index(self):
        lr1 = [60.0, 20.0, 5.0]
        cvt = [[13.4, 15.5, 19.9]] * 3
        df = trace_stats(_mock_trace(lr1, cvt))
        assert list(df.index) == ['r=0', 'r<=1', 'r<=2']


# ── eigen_stats ───────────────────────────────────────────────────────────────

def _mock_eigen(lr2, cvm):
    m = MagicMock()
    m.lr2 = lr2
    m.cvm = cvm
    return m


class TestEigenStats:
    _lr2 = [45.0, 8.0]
    _cvm = [[12.3, 14.1, 18.5], [2.7, 3.8, 6.6]]

    def test_columns(self):
        df = eigen_stats(_mock_eigen(self._lr2, self._cvm))
        assert list(df.columns) == ['eigen', '10%', '5%', '1%']

    def test_index_labels(self):
        df = eigen_stats(_mock_eigen(self._lr2, self._cvm))
        assert list(df.index) == ['r=0', 'r<=1']

    def test_shape(self):
        assert eigen_stats(_mock_eigen(self._lr2, self._cvm)).shape == (2, 4)

    def test_eigen_stats_populated(self):
        df = eigen_stats(_mock_eigen(self._lr2, self._cvm))
        assert df.loc['r=0',  'eigen'] == pytest.approx(45.0)
        assert df.loc['r<=1', 'eigen'] == pytest.approx(8.0)

    def test_critical_values_populated(self):
        df = eigen_stats(_mock_eigen([45.0], [[12.3, 14.1, 18.5]]))
        assert df.loc['r=0', '10%'] == pytest.approx(12.3)
        assert df.loc['r=0',  '5%'] == pytest.approx(14.1)
        assert df.loc['r=0',  '1%'] == pytest.approx(18.5)


# ── coint_diagnostics ─────────────────────────────────────────────────────────

def _make_mr_mock(zscore):
    m = MagicMock()
    m.get_mr_level.return_value = 0.01
    m.get_mr_rate.return_value = 0.5
    m.get_half_life.return_value = 15.0
    m.get_current_level.return_value = 0.02
    m.get_current_zscore.return_value = zscore
    m.get_stdev.return_value = 0.003
    m.get_const_pvalue.return_value = 0.01
    m.get_series_pvalue.return_value = 0.02
    return m


def _make_position_df(notionals):
    return pd.DataFrame({
        'market convention notional': notionals,
        'PX_LAST': [100.0] * len(notionals),
    })


def _run_diagnostics(diag_inputs, zscore, notionals=None, **kwargs):
    if notionals is None:
        notionals = [10.0, -5.0]
    df, test = diag_inputs
    position_df = _make_position_df(notionals)

    with patch(f"{_MOD}.mr.MeanRevertingTimeSeries", return_value=_make_mr_mock(zscore)), \
         patch(f"{_MOD}.tst.last_daily_hist_normal_vol", return_value=0.002), \
         patch(f"{_MOD}.mr.compute_sharpe_ratio", return_value={'Sharpe Ratio': 1.5}), \
         patch(f"{_MOD}.tst.create_position", return_value=(position_df, 50000.0)), \
         patch(f"{_MOD}.RSIIndicator"):
        return coint_diagnostics(test, df, verbose=False, **kwargs)


class TestCointDiagnostics:
    _EXPECTED_KEYS = {
        'Half life', 'Rounded weights', 'What you should trade',
        'What you should trade USD amount', 'Current zscore',
        'Trace (5%)', 'Eigen (5%)', 'Trace (10%)', 'Eigen (10%)',
        'Johansen Series', '1mio 1SD in USD', 'PX_LAST',
        'Series stdev', 'MR level', 'Half life Sharpe Ratio', 'RSI 14',
    }

    def test_returned_dict_has_all_keys(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-0.5)
        assert self._EXPECTED_KEYS == set(result.keys())

    def test_positive_zscore_flips_trade_amounts(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=+2.0, notionals=[10.0, -5.0])
        assert result['What you should trade'] == [-10.0, 5.0]
        assert result['What you should trade USD amount'] == pytest.approx(-50000.0)

    def test_negative_zscore_keeps_trade_amounts(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-2.0, notionals=[10.0, -5.0])
        assert result['What you should trade'] == [10.0, -5.0]
        assert result['What you should trade USD amount'] == pytest.approx(50000.0)

    def test_zero_zscore_does_not_flip(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=0.0, notionals=[10.0, -5.0])
        assert result['What you should trade'] == [10.0, -5.0]

    def test_weights_rounded_to_8_decimals(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-0.5)
        for w in result['Rounded weights']:
            assert w == round(w, 8)

    def test_trade_amounts_rounded_to_4_decimals(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-0.5,
                                  notionals=[10.123456789, -5.987654321])
        for t in result['What you should trade']:
            assert t == round(t, 4)

    def test_half_life_taken_from_mr(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-0.5)
        assert result['Half life'] == pytest.approx(15.0)

    def test_current_zscore_in_result(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=1.23)
        assert result['Current zscore'] == pytest.approx(1.23)

    def test_johansen_series_is_pandas_series(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-0.5)
        assert isinstance(result['Johansen Series'], pd.Series)

    def test_1mio_1sd_scales_stdev(self, diag_inputs):
        result = _run_diagnostics(diag_inputs, zscore=-0.5)
        assert result['1mio 1SD in USD'] == pytest.approx(1e6 * 0.003)


def test_cointegration_johansen_test_weights_shape():
    """One weight per asset (column)."""
    df = make_cointegrated()
    result = johansen_test(df)
    assert len(result["weights"]) == df.shape[1]


def test_cointegration_johansen_test_first_weight_is_one():
    """norm_1st_eigvec() divides by evec[0][0], so weights[0] must always be 1."""
    result = johansen_test(make_cointegrated())
    assert np.isclose(result["weights"][0], 1.0)


def test_cointegration_johansen_test_weight_ratio_close_to_minus_1_over_a():
    """For y2 = a*y1 + noise the ratio w[1]/w[0] must be ≈ -1/a."""
    a = 2.0
    # Extra-strong cointegration and long series for a tight numerical check
    # rng = np.random.default_rng(0)
    df = make_cointegrated(n=2000, a=a, noise_scale=0.001)#, rng=rng)
    w = johansen_test(df)["weights"]
    assert abs(w[1] / w[0] - (-1.0 / a)) < 0.05


def test_cointegration_johansen_test_basket_is_stationary():
    """ The Johansen basket w·y must drift far less than y1 alone """
    a = 2.0
    df = make_cointegrated(a=a)
    w = johansen_test(df)["weights"]

    basket = df["y1"].values * w[0] + df["y2"].values * w[1]
    basket_range = basket.max() - basket.min()
    y1_range    = df["y1"].max() - df["y1"].min()
    assert basket_range < 0.1 * y1_range


def test_cointegration_johansen_test_detects_cointegration_at_5pct():
    """A clearly cointegrated pair must pass both trace and eigen tests at 5%."""
    result = johansen_test(make_cointegrated(n=2000, noise_scale=0.001))
    assert result["trace (5%)"]
    assert result["eigen (5%)"]


def test_cointegration_johansen_test_10pct_at_least_as_liberal_as_5pct():
    """If a series passes at 5% it must also pass at the looser 10% threshold."""
    result = johansen_test(make_cointegrated())
    if result["trace (5%)"]:
        assert result["trace (10%)"]
    if result["eigen (5%)"]:
        assert result["eigen (10%)"]


def test_cointegration_johansen_stats_fast_cointegrated_passes():
    """Clearly cointegrated series must pass all four thresholds."""
    # rng = np.random.default_rng(1)
    df = make_cointegrated(n=2000, noise_scale=0.001)
    res_jo = coint_johansen(df, 0, 1)
    trace_5, trace_10, eigen_5, eigen_10 = check_johansen_stats_fast(res_jo)
    assert trace_10, "trace test at 10% should pass for a clearly cointegrated pair"
    assert eigen_10, "eigen test at 10% should pass for a clearly cointegrated pair"


def test_cointegration_johansen_stats_fast_consistent_with_johansen_test():
    """check_johansen_stats_fast results must agree with the flags in johansen_test."""
    df = make_cointegrated()
    high_level = johansen_test(df)

    res_jo = coint_johansen(df, 0, 1)
    trace_5, trace_10, eigen_5, eigen_10 = check_johansen_stats_fast(res_jo)

    assert high_level["trace (5%)"]  == trace_5
    assert high_level["trace (10%)"] == trace_10
    assert high_level["eigen (5%)"]  == eigen_5
    assert high_level["eigen (10%)"] == eigen_10
