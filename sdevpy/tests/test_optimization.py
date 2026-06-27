import numpy as np
import pytest
from sdevpy.maths.optimization import (
    create_optimizer, create_bounds, SciPyOptimizer, MultiOptimizer, record_history
)


def quadratic(x):
    return (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2


def test_optimizer_create():
    opt = create_optimizer('SLSQP')
    assert isinstance(opt, SciPyOptimizer)
    with pytest.raises(ValueError):
        create_optimizer('QuantumAnnealer')


def test_optimizer_create_bounds():
    b = create_bounds([0.0, 0.0], [5.0, 5.0])
    assert b is not None


def test_optimizer_nelder_mead_minimize():
    opt = create_optimizer('Nelder-Mead')
    result = opt.minimize(quadratic, x0=[0.0, 0.0])
    assert result.fun < 1e-6


def test_optimizer_de_minimize():
    opt = create_optimizer('DE')
    bounds = create_bounds([0.0, 0.0], [5.0, 5.0])
    result = opt.minimize(quadratic, bounds=bounds)
    assert result.fun < 1e-4


def test_optimizer_with_ftol():
    opt = create_optimizer('COBYLA', ftol=1e-8)
    result = opt.minimize(quadratic, x0=[0.0, 0.0])
    assert result.fun < 1e-8


def test_optimizer_multi_optimizer_minimize():
    opt = MultiOptimizer(methods=['L-BFGS-B', 'DE'], mtol=1e-4)
    bounds = create_bounds([0.0, 0.0], [5.0, 5.0])
    result = opt.minimize(quadratic, x0=[0.0, 0.0], bounds=bounds)
    assert result.fun < 1e-4


def test_optimizer_record_history_counts_evals():
    @record_history(enabled=True)
    def f(x):
        return x[0] ** 2

    opt = create_optimizer('Powell')
    opt.minimize(f, x0=[1.0])
    assert len(f.history) > 0


def test_optimizer_record_history_stores_x_and_f():
    @record_history(enabled=True)
    def f(x):
        return x[0] ** 2

    f(np.array([3.0]))
    assert f.history[0]['x'][0] == 3.0
    assert f.history[0]['f'] == 9.0
