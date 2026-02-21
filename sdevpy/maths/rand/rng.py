""" Sobol RNG wrapping SciPy """
import numpy as np
import scipy.stats as sp

Ninv = sp.norm.ppf


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
        rng = Sobol(dim, scramble=True)
        sob = rng.normal(num_mc)
        gaussians_ = [sob[:,num_factors * idx:num_factors*(idx + 1)] for idx in range(num_steps)]
    else:
        raise ValueError("Unknown method: " + method)

    return np.asarray(gaussians_)


if __name__ == "__main__":
    DIM = 2
    SCRAMBLE = False
    OPTIMIZATION = None # None, 'random-cd', 'lloyd'
    RNG = Sobol(dim=DIM, scramble=SCRAMBLE, optimization=OPTIMIZATION)
    NUM_SIM = 10
    DRAW_METHOD = "Exact"
    uni = RNG.uniform(NUM_SIM) #, DRAW_METHOD)
    print(uni)

    qmcSob = sp.qmc.Sobol(d=DIM, scramble=SCRAMBLE, optimization=OPTIMIZATION)
    print("num gen: ", qmcSob.num_generated)
    uni = qmcSob.random(1)
    print(uni)
    print("num gen: ", qmcSob.num_generated)
    uni = qmcSob.random(3)
    print(uni)
    print("num gen: ", qmcSob.num_generated)
    uni = qmcSob.random(4)
    print(uni)
    print("num gen: ", qmcSob.num_generated)
