from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from tools.sabr import calculate_strikes


def myfunc(v):
    a0 = 0.5
    ainf = 2.0
    b = -0.01
    tau = 5.0
    return ainf + (b * v + a0 - ainf) * np.exp(-v / tau)


n_points = 5
x = np.linspace(0.0, 4.0, num=n_points)
y = myfunc(x)
spline = CubicSpline(x, y, bc_type='natural')

k = np.linspace(-1.0, 5.0, num=10)
func = myfunc(k)
inter = spline(k)

plt.ioff()
plt.plot(k, func, color='blue', label='Function')
plt.plot(k, inter, color='red', label='Interpolation')
plt.legend(loc='upper right')
plt.show()

