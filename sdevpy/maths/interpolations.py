""" Various interpolations and parametric forms """
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sdevpy.maths import constants


def create_interpolation(**kwargs):
    interp = kwargs.get('interp', 'linear')
    l_extrap = kwargs.get('l_extrap', 'builtin')
    r_extrap = kwargs.get('r_extrap', 'builtin')
    interpolator = create_interpolator(interp, **kwargs)
    l_extrapolator = create_extrapolator(interpolator, l_extrap)
    r_extrapolator = create_extrapolator(interpolator, r_extrap)
    interpolation = Interpolation(interpolator, l_extrapolator, r_extrapolator)
    return interpolation


def create_extrapolator(interpolator, type='builtin'):
    type_dn = type.lower()
    match type_dn:
        case 'none': return NoneExtrapolator()
        case 'builtin': return interpolator
        case 'flat': return FlatExtrapolator()
        case 'linear': return LinearInterpolator()
        case _: raise TypeError(f"Unknown extrapolator type: {type}")


def create_interpolator(type='linear', **kwargs):
    type_dn = type.lower()
    match type_dn:
        case 'step': return StepInterpolator(**kwargs)
        case 'linear': return LinearInterpolator(**kwargs)
        case 'cubicspline': return CubicSplineInterpolator(**kwargs)
        case 'bspline': return BSplineInterpolator(**kwargs)
        case _: raise TypeError(f"Unknown interpolator type: {type}")


class Interpolator(ABC):
    def __init__(self, **kwargs):
        self.eps = kwargs.get('eps', 100.0 * constants.FLOAT_EPS)
        print(self.eps)
        x_grid = kwargs.get('x_grid', None)
        y_grid = kwargs.get('y_grid', None)
        if (x_grid is None and y_grid is not None) or (x_grid is not None and y_grid is None):
            raise ValueError("Ambiguous data: either set both x and y or set neither")
        elif x_grid is not None and y_grid is not None:
            set_data(x_grid, y_grid)

    def set_data(self, x_grid, y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.initialize()

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def value(self, x):
        pass


class CubicSplineInterpolator(Interpolator):
    """ Cubic spline wrapping scipy.interpolate. Defaulting to natural.
        For the boundary conditions, bc_type can be
            * a tuple ((n, bl), (n, br)) where n is 1 or 2 and bl and br are the values of the
              corresponding derivatives on the left and right ends
            * either of: natural, clamped, not-a-knot, periodic
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bc_type = kwargs.get('bc_type', 'natural')
        self.interp = None

    def initialize(self):
        self.interp = spi.CubicSpline(self.x_grid, self.y_grid, bc_type=self.bc_type)

    def value(self, x):
        y = self.interp(x)
        return y


class BSplineInterpolator(Interpolator):
    """ B-spline wrapping scipy.interpolate. Defaulting to natural, degree 3.
        For the boundary conditions, bc_type can be
            * a tuple ((n, bl), (n, br)) where n is 1 or 2 and bl and br are the values of the
              corresponding derivatives on the left and right ends
            * either of: natural, clamped, not-a-knot, periodic
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.degree = kwargs.get('degree', 3)
        self.bc_type = kwargs.get('bc_type', 'not-a-knot')
        self.interp = None

    def initialize(self):
        self.interp = spi.make_interp_spline(self.x_grid, self.y_grid, bc_type=self.bc_type, k=self.degree)

    def value(self, x):
        y = self.interp(x)
        return y


class StepInterpolator(Interpolator):
    """ Wrapping numpy. direction can be left or right, defaulting to lef """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.direction = kwargs.get('direction', 'left').lower()
        if self.direction not in ['left', 'l', 'right', 'r']:
            raise RuntimeError(f"Unknown step interpolation direction: {self.direction}")

        self.direction == ('left' if self.direction == 'l' else self.direction)
        self.direction == ('right' if self.direction == 'r' else self.direction)

    def initialize(self):
        pass

    def value(self, x):
        if self.direction == 'left':
            indices = np.searchsorted(self.x_grid, x, side='right')
            print(indices)
            indices = np.clip(indices, 0, len(self.y_grid) - 1)
            print(indices)
            y = [self.y_grid[i] for i in indices]
        elif self.direction == 'right':
            indices = np.searchsorted(self.x_grid, x, side='left') - 1
            np.clip(indices, 0, len(self.y_grid) - 1)
            y = [self.y_grid[i] for i in indices]
        else:
            raise RuntimeError(f"Unknown step interpolation direction(2): {self.direction}")

        if len(y) == 1:
            return y[0]
        else:
             return y



class LinearInterpolator(Interpolator):
    """ Wrapper around numpy's linear interpolation """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        self.xl = self.x_grid[0]
        self.xr = self.x_grid[-1]
        self.yl = self.y_grid[0]
        self.yr = self.y_grid[-1]
        self.cl = (self.y_grid[1] - self.y_grid[0]) / (self.x_grid[1] - self.xl)
        self.cr = (self.y_grid[-2] - self.y_grid[-1]) / (self.x_grid[-2] - self.xr)

    def value(self, x):
        x = np.asarray(x)
        v = np.interp(x, self.x_grid, self.y_grid)

        below = (x < self.xl + self.eps)
        above = (x > self.xr - self.eps)

        v = np.where(below, self.yl + self.cl * (x - self.xl), v)
        v = np.where(above, self.yr + self.cr * (x - self.xr), v)

        return v


######## Extrapolators ############################################################################
class NoneExtrapolator(Interpolator):
    """ No extrapolation. Error returned if used. """
    def value(x):
        raise ValueError("Extrapolation not allowed")


class FlatExtrapolator(Interpolator):
    """ Flat extrapolation on both sides """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        self.xl = self.x_grid[0]
        self.xr = self.x_grid[-1]
        self.yl = self.y_grid[0]
        self.yr = self.y_grid[-1]

    def value(self, x):
        v = [None] * len(x)
        below = (x < self.xl + self.eps)
        above = (x > self.xr - self.eps)

        v = np.where(below, self.yl, v)
        v = np.where(above, self.yr, v)

        return v


######## Interpolation ############################################################################
class Interpolation:
    def __init__(self, interp, l_extrap, r_extrap, **kwargs):
        self.eps = kwargs.get('eps', 100.0 * constants.FLOAT_EPS)
        self.l_extrap = l_extrap
        self.interp = interp
        self.r_extrap = r_extrap
        x_grid = kwargs.get('x_grid', None)
        y_grid = kwargs.get('y_grid', None)
        if (x_grid is None and y_grid is not None) or (x_grid is not None and y_grid is None):
            raise ValueError("Ambiguous data: either set both x and y or set neither")
        elif x_grid is not None and y_grid is not None:
            set_data(x_grid, y_grid)

    def set_data(self, x_grid, y_grid):
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.l_extrap.set_data(x_grid, y_grid)
        self.interp.set_data(x_grid, y_grid)
        self.r_extrap.set_data(x_grid, y_grid)

    def value(self, x):
        x = np.asarray(x)
        v = self.interp.value(x)
        below = (x < self.x_grid[0] + self.eps)
        above = (x > self.x_grid[-1] - self.eps)

        v = np.where(below, self.l_extrap.value(x), v)
        v = np.where(above, self.r_extrap.value(x), v)
        return v


if __name__ == "__main__":
    # Sample points
    x = [0, 1, 2, 3, 4]
    y = [0.5, 0.8, 1.7, 1.4, 1.1]

    # Define interpolations
    interps = []
    interp = create_interpolation() # Default linear
    interps.append(['linear', interp])
    # interp = create_interpolation(interp='bspline', l_extrap='flat', r_extrap='flat', degree=2)
    interp = create_interpolation(interp='step', l_extrap='flat', r_extrap='flat',
                                  direction='left')
    interps.append(['i2', interp])

    for interp in interps:
        interp[1].set_data(x, y)

    # Calculate
    n_points = 100
    x_calc = np.linspace(-1, 5, n_points)
    x_calc = np.append(x_calc, x)
    x_calc = sorted(set(x_calc))
    interp_data = []
    for interp in interps:
        interp_data.append(interp[1].value(x_calc))

    print(interps[1][1].value([0.0, 0.5, 1.0, 1.5, 4.0, 4.5]))

    # Plot
    plt.scatter(x, y, color='black', label='data')
    for i in range(len(interps)):
        plt.plot(x_calc, interp_data[i], label=interps[i][0])
    plt.legend(loc='upper right')
    plt.show()
