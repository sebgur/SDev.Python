import numpy as np
from sdevpy.maths.specialfunctions import rebonato


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
