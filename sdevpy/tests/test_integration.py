import numpy as np
from sdevpy.maths.integration import check_trapezoid



def test_linear_function():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    assert np.isclose(check_trapezoid(x, y), 2.0)


def test_gaussian_integrates_to_one():
    from scipy.stats import norm
    vol, t, n_dev = 0.20, 5.5, 5
    stdev = vol * np.sqrt(t)
    x = np.linspace(-n_dev * stdev, n_dev * stdev, 1000)
    y = norm.pdf(x, 0.0, stdev)
    assert np.isclose(check_trapezoid(x, y), 1.0, atol=1e-6)


def test_matches_numpy_trapezoid():
    x = np.linspace(0, np.pi, 500)
    y = np.sin(x)
    assert np.isclose(check_trapezoid(x, y), np.trapezoid(y, x), atol=1e-10)
