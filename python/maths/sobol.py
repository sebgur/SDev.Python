""" Sobol RNG wrapping SciPy """
import math
import numpy as np
import scipy.stats as sp

N = sp.norm.cdf
Ninv = sp.norm.ppf


class Sobol:
    """ Wrapper for Sobol class using SciPy """
    def __init__(self, dim, scramble=True, optimization=None):
        self.dim = dim
        self.scramble = scramble
        self.optimization = optimization
        self.sampler = sp.qmc.Sobol(d=dim, scramble=scramble, optimization=optimization)

    def uniform(self, num_draws, draw_method='Exact'):
        """ Draw num_draws uniforms. Different drawing methods:
            * Exact: draw exactly the requested number
            * Power: draw 2**power - 1 numbers for chosen power=num_draws
            * Floor: draw the largest number 2**power - 1 below the requested number
            * Ceil: draw the smallest number 2**power - 1 above the requested number """
        if draw_method == 'Power':
            raw = self.sampler.random_base2(m=num_draws)
        else:
            log = np.log(num_draws + 1) / np.log(2)
            if draw_method == 'Floor':
                power = math.floor(log)
                raw = self.sampler.random_base2(m=power)
            elif draw_method in ('Ceil', 'Exact'):
                power = math.ceil(log)
                raw = self.sampler.random_base2(m=power)
                if draw_method == 'Exact':
                    raw = raw[:num_draws + 1]
            else:
                raise ValueError("Invalid draw method: " + draw_method)

        # Remove first point as it's 0 (if not scrambled) and can't be transformed into normal
        clipped = raw[1:]
        return clipped

    def normal(self, num_sim, draw_method='Exact'):
        """ Draw normally distributed numbers, converted from uniforms using the normal inverse
            CDF """
        uniforms = self.uniform(num_sim, draw_method)
        return Ninv(uniforms)


if __name__ == "__main__":
    DIM = 2
    SCRAMBLE = False
    OPTIMIZATION = None # None, 'random-cd', 'lloyd'
    rng = Sobol(dim=DIM, scramble=SCRAMBLE, optimization=OPTIMIZATION)
    NUM_SIM = 4
    DRAW_METHOD = "Exact"
    uni = rng.uniform(NUM_SIM, DRAW_METHOD)
    print(uni)
