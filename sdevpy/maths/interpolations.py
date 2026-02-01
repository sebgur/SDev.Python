""" Various interpolations and parametric forms """
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import enum
from sdevpy.maths import constants


############ TODO ###################
# * Wrap all from scipy
# * Get monotone convex

def get(type, **kwargs):
    return 0


class FlatExtrapolator(Interpolator):
    """ Flat extrapolation on both sides """
    def __init__(self, **kwargs):
        super.__init__(kwargs)

    def initialize(self, x_grid, y_grid):
        self.xl = self.x_grid[0]
        self.xr = self.x_grid[-1]
        self.yl = self.y_grid[0]
        self.yr = self.y_grid[-1]

    def value(x):
        if x < xl + self.eps:
            return self.yl
        elif x > xr - self.eps:
            return self.yr
        else:
            raise ValueError("Flat extrapolator called within interpolation range")


class LinearInterpolator(Interpolator):
    """ Wrapper around numpy's linear interpolation """
    def __init__(self, **kwargs):
        super.__init__(kwargs)

    def initialize(self):
        self.yl = self.y_grid[0]
        self.yr = self.y_grid[-1]

    def value(x):
        v = np.interp(x, self.x_grid, self.y_grid, left=self.yl, right=self.yr)
        return v


class Interpolator(ABC):
    def __init__(self, **kwargs):
        self.eps = 100.0 * constants.FLOAT_EPS
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
    def value(x):
        pass


class Interpolation:
    def __init__(self, l_extrap, interp, r_extrap):
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
        l_extrap.set_data(x_grid, y_grid)
        interp.set_data(x_grid, y_grid)
        r_extrap.set_data(x_grid, y_grid)

    def value(x):
        if x <= x_grid[0]:
            return l_extrap.value(x)
        elif x >= x_grid[-1]:
            return r_extrap.value(x)
        else:
            return interp.value(x)


class extrap(Enum):
    NONE
    FLAT
    LINEAR
    BUILTIN


def rebonato(v):
    """ Parametric funtion in Rebonato's book, convenient to fit yield and volatility
        curves """
    a0 = 0.5
    ainf = 2.0
    b = -0.01
    tau = 5.0
    return ainf + (b * v + a0 - ainf) * np.exp(-v / tau)


if __name__ == "__main__":
    a = [1, 2, 3]
    print(a)
    b = a
    b[0] = 0
    print(a)
    print(b)

    # Sample points
    NUM_POINTS = 5
    x = np.linspace(0.0, 4.0, num=NUM_POINTS)
    y = rebonato(x)

    # Interpolations
    spline = CubicSpline(x, y, bc_type='natural')

    k = np.linspace(-1.0, 5.0, num=100)
    func = rebonato(k)
    scipy_interp = spline(k)
    np_interp = np.interp(k, x, y, left=0.0, right=0.0)

    plt.ioff()
    plt.plot(k, func, color='blue', label='Function')
    plt.plot(k, scipy_interp, color='red', label='SciPy')
    plt.plot(k, np_interp, color='green', label='NumPy')
    plt.legend(loc='upper right')
    plt.show()
