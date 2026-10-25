import numpy as np
import pytest
import scipy.optimize as opt
from sdevpy.maths.leastsquares import LeastSquaresProblem, create_lsq_solver, ScalarLeastSquaresSolver


B = np.array([0.8, 1.5, -0.4])
RNG = np.random.default_rng(7)
A = RNG.normal(size=(8, 3))
Y = RNG.normal(size=8)


def linear(x):
    return A @ x - Y, A


def shift(x):
    """ r = x - B: min |x - B|^2, closest point to B """
    return x - B, np.eye(len(x))


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]]), np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]])


ALL_METHODS = ['trf', 'dogbox', 'lm', 'SLSQP', 'trust-constr', 'L-BFGS-B', 'BFGS']
TIGHT = {'SLSQP': dict(ftol=1e-14), 'BFGS': dict(gtol=1e-10), 'trust-constr': dict(gtol=1e-12, xtol=1e-12, options={'initial_barrier_parameter': 1e-4}),
         'L-BFGS-B': dict(ftol=1e-15, gtol=1e-12)}  # SciPy default stopping criteria are loose


def solver(method):
    return create_lsq_solver(method, **TIGHT.get(method, {}))


@pytest.mark.parametrize("method", ALL_METHODS)
def test_linear_problem(method):
    res = solver(method).solve(LeastSquaresProblem(linear, np.zeros(3)))
    expected = np.linalg.lstsq(A, Y, rcond=None)[0]
    np.testing.assert_allclose(res.x, expected, atol=1e-6)
    assert res.success
    assert res.cost == pytest.approx(0.5 * np.sum((A @ res.x - Y) ** 2))
    np.testing.assert_allclose(res.residuals, A @ res.x - Y)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_rosenbrock(method):
    res = solver(method).solve(LeastSquaresProblem(rosenbrock, np.array([-1.2, 1.0])))
    np.testing.assert_allclose(res.x, [1.0, 1.0], atol=1e-5)


@pytest.mark.parametrize("method", ['trf', 'dogbox', 'SLSQP', 'trust-constr', 'L-BFGS-B'])
def test_bounds(method):
    bounds = opt.Bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    res = solver(method).solve(LeastSquaresProblem(shift, np.full(3, 0.5), bounds=bounds))
    np.testing.assert_allclose(res.x, np.clip(B, 0.0, 1.0), atol=1e-6)


@pytest.mark.parametrize("method, tol", [('SLSQP', 1e-6), ('trust-constr', 1e-6), ('COBYLA', 1e-3)])
def test_linear_constraint(method, tol):
    """ sum(x) <= 1: projection of B on the half-space """
    con = opt.LinearConstraint(np.ones((1, 3)), -np.inf, 1.0)
    res = solver(method).solve(LeastSquaresProblem(shift, np.zeros(3), constraints=[con]))
    np.testing.assert_allclose(res.x, B - (B.sum() - 1.0) / 3.0, atol=tol)


@pytest.mark.parametrize("method", ['SLSQP', 'trust-constr'])
def test_nonlinear_constraint(method):
    """ |x|^2 <= 1: projection of B on the unit ball """
    con = opt.NonlinearConstraint(lambda x: x @ x, -np.inf, 1.0, jac=lambda x: 2.0 * x[None, :])
    res = solver(method).solve(LeastSquaresProblem(shift, np.zeros(3), constraints=[con]))
    np.testing.assert_allclose(res.x, B / np.linalg.norm(B), atol=1e-6)


def test_badly_scaled_problem():
    """ Residual sensitivities ~1e6 (bp residuals on rate-like unknowns): needs x_scale for SLSQP """
    def scaled(x):
        return 1e6 * (x - 0.01 * B), 1e6 * np.eye(3)
    res = create_lsq_solver('SLSQP').solve(LeastSquaresProblem(scaled, np.zeros(3)))
    np.testing.assert_allclose(res.x, 0.01 * B, atol=1e-10)
    res = ScalarLeastSquaresSolver('SLSQP', x_scale=1e6).solve(LeastSquaresProblem(scaled, np.zeros(3)))
    np.testing.assert_allclose(res.x, 0.01 * B, atol=1e-10)


def test_one_evaluation_per_point():
    calls = []
    def counted(x):
        calls.append(x.copy())
        return linear(x)
    res = create_lsq_solver('trf').solve(LeastSquaresProblem(counted, np.zeros(3)))
    assert res.nfev == len(calls)
    assert len({tuple(c) for c in calls}) == len(calls)


def test_unsupported_combinations():
    con = opt.LinearConstraint(np.ones((1, 3)), -np.inf, 1.0)
    bounds = opt.Bounds(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError):
        create_lsq_solver('trf').solve(LeastSquaresProblem(shift, np.zeros(3), constraints=[con]))
    with pytest.raises(ValueError):
        create_lsq_solver('lm').solve(LeastSquaresProblem(shift, np.zeros(3), bounds=bounds))
    with pytest.raises(ValueError):
        create_lsq_solver('L-BFGS-B').solve(LeastSquaresProblem(shift, np.zeros(3), constraints=[con]))
    with pytest.raises(ValueError):
        create_lsq_solver('bogus')
