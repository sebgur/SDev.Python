""" Optimization """
# https://machinelearningmastery.com/how-to-use-nelder-mead-optimization-in-python/
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# For DE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
from abc import abstractmethod
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sdevpy.maths.constants import FLOAT_MAX


SCIPY_OPTIMIZERS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA',
                    'SLSQP', 'trust-constr', 'Newton-CG', 'dogleg', 'trust-ncg',
                    'trust-exact', 'trust-krylov', 'DE']


# Available strategies for DE
# best1bin, best1exp, best2exp, best2bin, rand1bin, rand1exp, rand2bin, rand2exp,
# randtobest1bin, randtobest1exp, currenttobest1bin, currenttobest1exp

def create_optimizer(method, **kwargs):
    """ Create an optimizer. Currently only supporting SciPy """
    optimizer = None
    if method in SCIPY_OPTIMIZERS:
        optimizer = SciPyOptimizer(method, **kwargs)
    else:
        raise RuntimeError("Optimizer type not supported: " + method)

    return optimizer


def create_bounds(lw_bounds, up_bounds):
    return opt.Bounds(lw_bounds, up_bounds, keep_feasible=False)


class Optimizer:
    @abstractmethod
    def minimize(self, f, x0, args, bounds):
        """ Minimization """


class SciPyOptimizer(Optimizer):
    """ Wrapper for SciPy optimizers, including differential_evolution """
    def __init__(self, method = 'Powell', **kwargs):
        self.method_ = method
        self.std_minimizers = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA',
                               'SLSQP', 'trust-constr', 'Newton-CG', 'dogleg', 'trust-ncg',
                               'trust-exact', 'trust-krylov']

        self.other_minimizers = ['DE']
        self.kwargs = kwargs
        self.options = {}
        ftol = kwargs.get('ftol', None)
        xtol = kwargs.get('xtol', None)
        gtol = kwargs.get('gtol', None)
        if ftol is not None: # Tolerance on objective
            self.options['ftol'] = ftol
        if xtol is not None: # Tolerance on parameters
            self.options['xtol'] = xtol
        if gtol is not None: # Tolerance on gradient
            self.options['gtol'] = gtol

        if self.method_ not in self.std_minimizers and self.method_ not in self.other_minimizers:
            raise RuntimeError("Method " + self.method_ + " not found in SciPy")

    def minimize(self, f, x0=None, args=(), bounds=None):
        result = None
        if self.method_ in self.std_minimizers:
            tol = self.kwargs.get('tol', None)
            result = opt.minimize(f, x0, args, method=self.method_, bounds=bounds,
                                  tol=tol, options=self.options)
        elif self.method_== 'DE':
            atol = self.kwargs.get('atol', 0)
            popsize = self.kwargs.get('popsize', 15)
            strategy = self.kwargs.get('strategy', 'best1bin')
            recombination = self.kwargs.get('recombination', 0.7)
            mutation = self.kwargs.get('mutation', (0.5, 1.0)) # ToDo: parameter not used
            result = opt.differential_evolution(f, x0=x0, args=args, bounds=bounds, atol=atol,
                                                popsize=popsize, strategy=strategy,
                                                recombination=recombination)
        else:
            raise RuntimeError("Method " + self.method_ + " not recognized")

        return result


class MultiOptimizer(Optimizer):
    """ Wrapper for SciPy optimizers, including differential_evolution """
    def __init__(self, methods = ['L-BFGS-B', 'DE'], mtol=1e-4, **kwargs):
        self.methods_ = methods
        self.mtol_ = mtol
        self.kwargs = kwargs

        self.optimizers_ = []
        for method in self.methods_:
            self.optimizers_.append(create_optimizer(method, **kwargs))

    def minimize(self, f, x0=None, args=(), bounds=None):
        result = None
        nfev = 0
        for i, optimizer in enumerate(self.optimizers_):
            print("Trying optimization using " + self.methods_[i] + ": ", end='')
            result = optimizer.minimize(f, x0, args, bounds)
            nfev = nfev + result.nfev
            if result.fun < self.mtol_:
                print("Good enough!")
                break
            elif i < len(self.methods_) - 1:
                print("Continuing")
            else:
                print("Stopping")

        return result, nfev


def record_history(enabled=True, verbose=False):
    """ Decorator that records the history of evaluation of the given function """
    def decorator(func):
        # Initialize history
        recorder = {'enabled': enabled}
        history = []
        def wrapped(x, *args, **kwargs):
            result = func(x, *args, **kwargs)

            if not recorder['enabled']:
                return result

            # Record
            record = {'eval': len(history) + 1, 'x': x.copy(), 'f': result}
            history.append(record)

            if verbose:
                print(f"Eval {record['eval']}, Point = {x}, Value = {record['f']}")

            return result

        wrapped.recorder = recorder
        wrapped.history = history
        return wrapped

    return decorator


if __name__ == "__main__":
    # Objective function
    @record_history(enabled=True, verbose=False)
    def f(x, *args):
        x_ = x[0]
        a = args[0]
        b = args[1]
        c = args[2]
        prod = a * b * c
        bi = a * b + b * c + a * c
        sum = a + b + c
        value = 0.25 * np.power(x_, 4) - sum / 3.0 * np.power(x_, 3) + 0.5 * bi * x_**2 - prod * x_ + 1
        return value


    # Choose method
    # method = 'Nelder-Mead'
    # method = "Powell" # Success x^4
    # method = "CG"
    # method = "BFGS"
    # method = "L-BFGS-B"
    # method = "TNC"
    # method = "COBYLA" # Success x^4
    method = "SLSQP"
    # method = "trust-constr"
    # method = "Newton-CG" # Requires Jacobian
    # method = "dogleg" # Requires Jacobian
    # method = "trust-ncg" # Requires Jacobian
    # method = "trust-exact" # Requires Jacobian
    # method = "trust-krylov" # Requires Jacobian
    # method = "DE" # Success x^4

    # Create the optimizer
    optimizer = create_optimizer(method, ftol=0.0001)

    # Define the bounds
    bounds = opt.Bounds([0], [4], keep_feasible=False)

    # Define initial point
    init_point = 2.5

    # Optimize
    result = optimizer.minimize(f, x0=[init_point], args=(1, 2, 3.2), bounds=bounds)
    x = result.x
    fun = result.fun

    # Pring results
    print("Sol Point", x)
    print("Sol Value", fun)
    for key in result.keys():
        if key in result:
            print(f"{key}: {result[key]}")

    # Plot solution
    points = np.linspace(0, 4, 100).reshape(-1, 1)
    y = []
    f.recorder['enabled'] = False
    for p in points:
        y.append(f(p, 1, 2, 3.2))

    plt.plot(points, y, color='blue', alpha=0.8, label='Objective')
    plt.scatter(x[0], fun, color='red', alpha=1.0, label='Solution')
    plt.show()

    # Plot history
    history = f.history
    xx = [h['eval'] for h in history]
    ff = [h['f'] for h in history]
    plt.plot(xx, ff)
    plt.show()
