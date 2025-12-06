""" Various interpolations and parametric forms """
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def rebonato(v):
    """ Parametric funtion in Rebonato's book, convenient to fit yield and volatility
        curves """
    a0 = 0.5
    ainf = 2.0
    b = -0.01
    tau = 5.0
    return ainf + (b * v + a0 - ainf) * np.exp(-v / tau)


if __name__ == "__main__":
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
