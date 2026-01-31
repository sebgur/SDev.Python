""" Various interpolations and parametric forms """
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import enum


############ TODO ###################
# * Get linear from numpy
# * Wrap all from scipy
# * Get monotone convex

def get(type, **kwargs):
    return 0


def rebonato(v):
    """ Parametric funtion in Rebonato's book, convenient to fit yield and volatility
        curves """
    a0 = 0.5
    ainf = 2.0
    b = -0.01
    tau = 5.0
    return ainf + (b * v + a0 - ainf) * np.exp(-v / tau)


class Interpolator(ABC):
    def __init__(self, **kwargs):
        self.x_grid = kwargs.get('x_grid', None)
        self.y_grid = kwargs.get('y_grid', None)
        self.interp = None
        self.l_extrap = extrap.FLAT
        self.r_extrap = extrap.FLAT

    def set_data(self, x_grid, y_grid):
        pass

    @abstractmethod
    def interpolate(x):
        pass


class Extrapolator:


class extrap(Enum):
    NONE
    FLAT
    LINEAR
    BUILTIN


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
