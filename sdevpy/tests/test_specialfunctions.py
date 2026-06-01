import numpy as np
from sdevpy.maths.specialfunctions import rebonato
from sdevpy.montecarlo.smoothers import approx_cdf, smooth_max_diff


def test_rebonato():
    """ rebonato(0) must equal a0 exactly """
    assert rebonato(0.0) == 0.5

    """ rebonato(5) = 2 + (-0.05 - 1.5)*exp(-1) """
    expected = 2.0 + (-0.01 * 5.0 + 0.5 - 2.0) * np.exp(-1.0)
    assert abs(rebonato(5.0) - expected) < 1e-15

    """ As v → ∞ the result approaches ainf = 2.0 """
    assert abs(rebonato(1e6) - 2.0) < 1e-10

    """ Accepts a NumPy array and returns same shape """
    v = np.array([0.0, 5.0, 10.0, 100.0])
    result = rebonato(v)
    assert result.shape == v.shape

    """ Vectorized output matches element-wise scalar calls """
    v = np.array([0.0, 1.0, 5.0, 10.0])
    expected = np.array([rebonato(vi) for vi in v])
    assert np.allclose(rebonato(v), expected)


def test_approx_cdf():
    assert np.isclose(approx_cdf(0.0), 0.5)
    assert approx_cdf(100.0) > 0.999
    assert approx_cdf(-100.0) < 0.001


def test_smooth_max_diff():
    # spot >> strike: smoothed call ≈ spot - strike
    result = smooth_max_diff(10000.0, 100.0)
    assert np.isclose(result, 10000.0 - 100.0, rtol=1e-4)
    # spot << strike: smoothed call ≈ 0
    result = smooth_max_diff(1.0, 100.0)
    assert abs(result) < 1e-6
    spots = np.array([80.0, 100.0, 120.0])
    result = smooth_max_diff(spots, 100.0)
    assert result.shape == (3,)
