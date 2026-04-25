import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sdevpy.timeseries.cointegration import johansen_test, check_johansen_stats_fast


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
