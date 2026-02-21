import numpy as np
from scipy.stats import qmc
from scipy.stats import norm


class Sobol:
    """ Wrapper for the Sobol class in scipy's qmc """
    def __init__(self, dim, **kwargs):
        self.dim = dim
        scramble = kwargs.get('scramble', True)
        # Not sure what this parameter is
        optimization = kwargs.get('optimization', None) # None, 'random-cd', 'lloyd'
        self.sampler = qmc.Sobol(d=dim, scramble=scramble, optimization=optimization)
        self.sampler.random(1) # Skip the first item as it is exact 0

    def uniform(self, n_draws):
        """ Draw n_draws uniforms """
        return self.sampler.random(n_draws)

    def normal(self, n_draws):
        """ Draw n_draws gaussians, converted from uniforms with scipy's normal inverse CDF """
        uniforms = self.uniform(n_draws)
        return norm.ppf(uniforms)

    def n_generated(self):
        """ Number of sequences generated (not counting the first pure 0) """
        return self.sampler.num_generated - 1 # Removing the 0


if __name__ == "__main__":
    DIM = 2
    SCRAMBLE = False
    OPTIMIZATION = None

    rng = Sobol(dim=DIM, scramble=SCRAMBLE, optimization=OPTIMIZATION)
    gaussians = rng.normal(10)
    print(gaussians)
    print(rng.n_generated())
