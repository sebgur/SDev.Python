""" Optimization """
# https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Define the function

def f(x, *args):
    x_ = x[0]
    a = args[0]
    b = args[1]
    c = args[2]
    prod = a * b * c
    bi = a * b + b * c + a * c
    sum = a + b + c
    return 0.25 * np.power(x_, 4) - sum / 3.0 * np.power(x_, 3) + 0.5 * bi * x_**2 - prod * x_ + 1



# method = 'Nelder-Mead'
method = "Powell" # Success x^4
# method = "CG"
# method = "BFGS"
# method = "L-BFGS-B"
# method = "TNC"
# method = "COBYLA" # Success x^4
# method = "SLSQP"
# method = "trust-constr"
# method = "Newton-CG" # Requires Jacobian
# method = "dogleg" # Requires Jacobian
# method = "trust-ncg" # Requires Jacobian
# method = "trust-exact" # Requires Jacobian
# method = "trust-krylov" # Requires Jacobian

# # Minimize the function
bounds = opt.Bounds([0], [4], keep_feasible=False)

# result = opt.minimize(f, x0=[1.5], args=(1, 2, 3.2), method=method, bounds=bounds)

result = opt.differential_evolution(f, x0=[1.5], args=(1, 2, 3.2), bounds=bounds)

print(result.message)

# print(result.fun)

print(result.x)


points = np.linspace(0, 4, 100).reshape(-1, 1)
# print(points)
y = []
for p in points:
    y.append(f(p, 1, 2, 3.2))

# plt.plot(points, y, color='blue', alpha=0.8, label='CF')
# plt.show()

