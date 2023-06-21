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
    NUM_POINTS = 5
    x = np.linspace(0.0, 4.0, num=NUM_POINTS)
    y = rebonato(x)
    spline = CubicSpline(x, y, bc_type='natural')

    k = np.linspace(-1.0, 5.0, num=10)
    func = rebonato(k)
    inter = spline(k)

    plt.ioff()
    plt.plot(k, func, color='blue', label='Function')
    plt.plot(k, inter, color='red', label='Interpolation')
    plt.legend(loc='upper right')
    plt.show()
