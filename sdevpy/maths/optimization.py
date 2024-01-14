""" Optimization """
# https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

from abc import abstractmethod
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


SCIPY_OPTIMIZERS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA',
                    'SLSQP', 'trust-constr', 'Newton-CG', 'dogleg', 'trust-ncg',
                    'trust-exact', 'trust-krylov', 'DE']


def create_optimizer(method):
    """ Create an optimizer. Currently only supporting SciPy """
    optimizer = None
    if method in SCIPY_OPTIMIZERS:
        optimizer = SciPyOptimizer(method)
    else:
        raise RuntimeError("Optimizer type not supported: " + method)
    
    return optimizer


class Optimizer:
    # def __init__(self):

    @abstractmethod
    def minimize(self, f, x0, args, bounds):
        """ Minimization """


class SciPyOptimizer(Optimizer):
    """ Wrapper for SciPy optimizers, including differential_evolution """
    def __init__(self, method = 'Powell'):
        self.method_ = method
        self.std_minimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA',
                                'SLSQP', 'trust-constr', 'Newton-CG', 'dogleg', 'trust-ncg',
                                'trust-exact', 'trust-krylov']

        self.other_minimizers = ['DE']
        
        if self.method_ not in self.std_minimizers and self.method_ not in self.other_minimizers:
            raise RuntimeError("Method " + self.method_ + " not found in SciPy")

    def minimize(self, f, x0, args, bounds):
        result = None
        if self.method_ in self.std_minimizers:
            result = opt.minimize(f, x0, args, method=self.method_, bounds=bounds)
        elif self.method_== 'DE':
            result = opt.differential_evolution(f, x0=x0, args=args, bounds=bounds)
        else:
            raise RuntimeError("Method " + self.method_ + " not recognized")
        
        return result


if __name__ == "__main__":
    # Objective function
    def f(x, *args):
        x_ = x[0]
        a = args[0]
        b = args[1]
        c = args[2]
        prod = a * b * c
        bi = a * b + b * c + a * c
        sum = a + b + c
        return 0.25 * np.power(x_, 4) - sum / 3.0 * np.power(x_, 3) + 0.5 * bi * x_**2 - prod * x_ + 1


    # Choose method
    method = 'Nelder-Mead'
    # method = "Powell" # Success x^4
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
    # method = "DE" # Success x^4

    # Create the optimizer
    optimizer = create_optimizer(method)

    # Define the bounds
    bounds = opt.Bounds([0], [4], keep_feasible=False)

    # Optimize
    result = optimizer.minimize(f, x0=[1.5], args=(1, 2, 3.2), bounds=bounds)

    for key in result.keys():
        if key in result:
            print(key + "\n", result[key])

    x = result.x
    fun = result.fun
    # print("Keys\n", result.keys())

    # Plot
    points = np.linspace(0, 4, 100).reshape(-1, 1)
    y = []
    for p in points:
        y.append(f(p, 1, 2, 3.2))

    plt.plot(points, y, color='blue', alpha=0.8, label='Objective')
    plt.scatter(x[0], fun, color='red', alpha=1.0, label='Solution')
    plt.show()
