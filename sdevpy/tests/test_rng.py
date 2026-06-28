import numpy as np
import pytest
from sdevpy.maths.rand.rng import gaussians, get_rng, MersenneTwister, Sobol, Halton, LatinHypercube


N, DIM = 64, 3


def test_rng_mt():
    rng = MersenneTwister(dim=DIM, seed=0)
    u = rng.uniform(N)
    assert u.shape == (N, DIM)
    assert np.all(u >= 0.0) and np.all(u <= 1.0)

    g = rng.normal(N)
    assert g.shape == (N, DIM)


def test_rng_sobol():
    rng = Sobol(dim=DIM, scramble=False)
    u = rng.uniform(N)
    assert u.shape == (N, DIM)
    assert rng.n_generated() == N


def test_rng_halton():
    rng = Halton(dim=DIM, scramble=False)
    u = rng.uniform(N)
    assert u.shape == (N, DIM)
    assert rng.n_generated() == N


def test_rng_latinhypercube():
    rng = LatinHypercube(dim=DIM, seed=0)
    u = rng.uniform(N)
    assert u.shape == (N, DIM)
    assert rng.n_generated() == N


def test_get_rng():
    rng = get_rng(dim=2, rng_type='MT')
    assert isinstance(rng, MersenneTwister)

    rng = get_rng(dim=2, rng_type='sobol')
    assert isinstance(rng, Sobol)

    rng = get_rng(dim=2, rng_type='halton')
    assert isinstance(rng, Halton)

    rng = get_rng(dim=2, rng_type='latinhypercube')
    assert isinstance(rng, LatinHypercube)

    with pytest.raises(TypeError):
        get_rng(dim=2, rng_type='unknown')


def test_rng_gaussians():
    g = gaussians(num_steps=5, num_mc=100, num_factors=2, method='PseudoRandom')
    assert g.shape == (5, 100, 2)

    g = gaussians(num_steps=4, num_mc=64, num_factors=2, method='Sobol')
    assert g.shape == (4, 64, 2)

    with pytest.raises(ValueError):
        gaussians(num_steps=2, num_mc=10, num_factors=1, method='unknown')
