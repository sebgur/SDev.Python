import numpy as np
from scipy.stats import norm
import time


def check_trapezoid(x, y):
    n = len(x)
    sum = 0.0
    for i in range(n - 1):
        sum += (y[i + 1] + y[i]) * (x[i + 1] - x[i]) / 2.0

    return sum


if __name__ == "__main__":
    n_points = 100
    vol = 0.20
    t = 5.5
    stdev = vol * np.sqrt(t)
    n_dev = 5
    x_max = stdev * n_dev
    x = np.linspace(-x_max, x_max, n_points)
    y = norm.pdf(x, 0.0, stdev)

    start = time.time()
    np_int = np.trapezoid(y, x)
    print("Numpy")
    print(np_int)
    print(time.time() - start)

    start = time.time()
    sd_int = check_trapezoid(x, y)
    print("In-house")
    print(sd_int)
    print(time.time() - start)

