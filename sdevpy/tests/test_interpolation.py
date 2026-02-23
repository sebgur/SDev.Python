import numpy as np
from sdevpy.maths import interpolation as itp

# Interpolation types: step, linear, cubicspline, bspline
# Extrapolation types: none, builtin, flat, linear

X_GRID = [0.0, 1.0, 2.0, 3.0]
Y_GRID = [10.0, 18.0, 25.0, 22.0]
TEST_X = [-0.5, 0.0, 0.2, 1.0, 1.4, 2.0, 2.7, 3.0, 3.5]


def test_linear_interpolation():
    interp = itp.create_interpolation()
    interp.set_data(X_GRID, Y_GRID)
    test = np.asarray(interp.value(TEST_X))
    ref = np.asarray([6, 10, 11.6, 18, 20.8, 25, 22.9, 22, 20.5])
    assert np.allclose(test, ref, 1e-12)


def test_cubicspline_interpolation():
    type = 'cubicspline'
    interp = itp.create_interpolation(interp=type, l_extrap='builtin', r_extrap='flat')
    interp.set_data(X_GRID, Y_GRID)
    test = np.asarray(interp.value(TEST_X))
    ref = np.asarray([6.15, 10, 11.5232, 18, 21.52, 25, 23.6098, 22, 22])
    assert np.allclose(test, ref, 1e-10)


def test_step_interpolation():
    type = 'step'
    interp = itp.create_interpolation(interp=type, l_extrap='flat', r_extrap='builtin')
    interp.set_data(X_GRID, Y_GRID)
    test = np.asarray(interp.value(TEST_X))
    ref = np.asarray([10, 10, 18, 25, 25, 22, 22, 22, 22])
    assert np.allclose(test, ref, 1e-10)


def test_bspline_interpolation():
    type = 'bspline'
    interp = itp.create_interpolation(interp=type, l_extrap='builtin', r_extrap='linear')
    interp.set_data(X_GRID, Y_GRID)
    test = np.asarray(interp.value(TEST_X))
    ref = np.asarray([8.4375, 10, 11.248, 18, 21.424, 25, 24.4855, 22, 20.5])
    assert np.allclose(test, ref, 1e-10)


if __name__ == "__main__":
    from sdevpy.tools import clipboard as clp
    type = 'bspline'
    interp = itp.create_interpolation(interp=type, l_extrap='builtin', r_extrap='linear')
    interp.set_data(X_GRID, Y_GRID)
    test = interp.value(TEST_X)
    print(test)
    # clp.export1d(test)
