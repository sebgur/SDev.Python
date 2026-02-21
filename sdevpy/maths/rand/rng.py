""" Sobol RNG wrapping SciPy """
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm, qmc


def gaussians(num_steps, num_mc, num_factors, method='PseudoRandom'):
    """ Easy-input method to draw gaussians in dimensions num_steps x num_mc x num_factors """
    gaussians_ = None
    if method == 'PseudoRandom':
        # Define dimensions
        mean = np.zeros(num_factors)
        corr = np.zeros((num_factors, num_factors))
        for i in range(num_factors):
            corr[i, i] = 1.0

        # Draw for each step
        seed = 42
        rng = np.random.RandomState(seed)
        gaussians_ = []
        for i in range(num_steps):
            gaussians_.append(rng.multivariate_normal(mean, corr, size=num_mc))
    elif method == 'Sobol':
        dim = num_steps * num_factors
        rng = Sobol(dim=dim, scramble=True)
        sob = rng.normal(num_mc)
        gaussians_ = [sob[:,num_factors * idx:num_factors*(idx + 1)] for idx in range(num_steps)]
    else:
        raise ValueError("Unknown method: " + method)

    return np.asarray(gaussians_)


def get_rng(dim=1, **kwargs):
    rng_type = kwargs.get('rng_type', 'MT')
    match rng_type.lower():
        case 'mt':
            seed = kwargs.get('seed', 42)
            return MersenneTwiser(dim=dim, seed=seed)
        case 'sobol':
            return Sobol(dim=dim, **kwargs)
        case _:
            raise TypeError(f"Unknown RNG type: {rng_type}")


class RandomNumberGenerator(ABC):
    def __init__(self, dim=1):
        self.dim = dim

    @abstractmethod
    def uniform(self, n_draws):
        pass

    def normal(self, n_draws):
        """ Draw n_draws gaussians, converted from uniforms with scipy's normal inverse CDF """
        uniforms = self.uniform(n_draws)
        return norm.ppf(uniforms)


class MersenneTwiser(RandomNumberGenerator):
    def __init__(self, dim=1, seed=42):
        super().__init__(dim)
        self.seed = seed
        self.mt = np.random.RandomState(self.seed)

    def uniform(self, n_draws):
        u = self.mt.uniform(size=(n_draws, self.dim))
        return u


class Sobol(RandomNumberGenerator):
    """ Wrapper for the Sobol class in scipy's qmc """
    def __init__(self, dim=1, **kwargs):
        super().__init__(dim)
        scramble = kwargs.get('scramble', True)
        # Not sure what this parameter is
        optimization = kwargs.get('optimization', None) # None, 'random-cd', 'lloyd'
        self.sampler = qmc.Sobol(d=dim, scramble=scramble, optimization=optimization)
        self.sampler.random(1) # Skip the first item as it is exact 0

    def uniform(self, n_draws):
        """ Draw n_draws uniforms """
        return self.sampler.random(n_draws)

    def n_generated(self):
        """ Number of sequences generated (not counting the first pure 0) """
        return self.sampler.num_generated - 1 # Removing the 0


if __name__ == "__main__":
    DIM = 2
    SCRAMBLE = False
    OPTIMIZATION = None

    rng1 = MersenneTwiser(dim=DIM, seed=42)
    gaussians = rng1.normal(10)
    print(gaussians)

    rng2 = Sobol(dim=DIM, scramble=SCRAMBLE, optimization=OPTIMIZATION)
    gaussians = rng2.normal(10)
    print(gaussians)
    print(rng2.n_generated())
