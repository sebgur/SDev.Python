import numpy as np
from sdevpy.models.multiasset_heston import MultiAssetHeston


def make_model():
    s0 = [100.0, 200.0]
    v0 = [0.04, 0.09]
    r = 0.02
    q = [0.01, 0.0]
    kappa = [2.0, 1.5]
    theta = [0.04, 0.09]
    xi = [0.3, 0.2]
    corr = [[1.0, 0.5, 0.2, 0.1],
            [0.5, 1.0, 0.1, 0.3],
            [0.2, 0.1, 1.0, 0.4],
            [0.1, 0.3, 0.4, 1.0]]
    return MultiAssetHeston(s0, v0, r, q, kappa, theta, xi, corr)


def test_step_prices_positive():
    m = make_model()
    n_paths = 500
    s, v = m.initial_state(n_paths)
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n_paths, m.dim))
    s_next, _ = m.step(s, v, 1.0 / 52, z)
    assert np.all(s_next > 0)


def test_step_variances_non_negative():
    m = make_model()
    n_paths = 500
    s, v = m.initial_state(n_paths)
    rng = np.random.default_rng(2)
    z = rng.standard_normal((n_paths, m.dim))
    _, v_next = m.step(s, v, 1.0 / 52, z)
    assert np.all(v_next >= 0)


def test_step_truncates_negative_variance():
    # Force v negative to exercise the full-truncation clamp
    m = make_model()
    n_paths = 10
    s = np.tile(m.s0, (n_paths, 1))
    v = np.full((n_paths, 2), -0.01)   # negative on entry
    z = np.zeros((n_paths, m.dim))
    s_next, v_next = m.step(s, v, 1.0 / 52, z)
    assert np.all(v_next >= 0)
    assert np.all(s_next > 0)
