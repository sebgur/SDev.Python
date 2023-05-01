""" Monte-Carlo simulation for XSABR models (vanillas) """
import numpy as np
from tools.timegrids import SimpleTimeGridBuilder


def prices(expiries, strikes, are_calls, fwd, shift, parameters, num_mc, steps_per_year):
    """ Calculate vanilla prices under XSABR model """
    # Build time grid
    time_grid_builder = SimpleTimeGridBuilder(steps_per_year=steps_per_year)
    time_grid_builder.add_grid(expiries)
    time_grid = time_grid_builder.complete_grid()
    num_steps = len(time_grid)
    print("time grid size\n", num_steps)
    print("time grid\n", time_grid)

    # Find payoff times
    is_payoff = np.in1d(time_grid, expiries)
    print("is_payoff\n", is_payoff)

    # Retrieve parameters
    scale = 0 # ToDo
    alpha = 0 # ToDo
    beta = parameters[1]
    nu = parameters[2]
    nu2 = nu**2
    rho = parameters[3]

    # Correlation matrix
    sqrtmrho2 = np.sqrt(1.0 - rho**2)
    corr = np.ones((2, 2))
    corr[0, 1] = corr[1, 0] = 0.0
    # corr[0, 1] = corr[1, 0] = parameters[3]
    print("corr\n", corr)

    # Set RNG
    means = np.zeros(2)
    seed = 42
    rng = np.random.RandomState(seed)

    # Initialize paths
    spot = np.ones(num_mc) * (fwd + shift)
    vol = np.ones(num_mc) * 1.0

    # Loop over time grid
    ts = te = 0
    for i, t in enumerate(time_grid):
        ts = te
        te = t
        dt = te - ts
        sqrt_dt = np.sqrt(dt)
        # print("ts\n", ts)
        # print("te\n", te)
        # print("dt\n", dt)
        # print("sqrt_dt\n", sqrt_dt)

        # Evolve
        gaussians = rng.multivariate_normal(means, corr, size=num_mc)
        dz0 = gaussians[:, 0] * sqrt_dt
        dz1 = gaussians[:, 1] * sqrt_dt
        print("gaussians\n", gaussians)
        print("dz0\n", dz0)
        print("dz1\n", dz1)

        # Scheme
        avolc = alpha * np.min(np.power(spot, beta - 1.0), 500.0 * np.power(scale, beta - 1.0))
        vols = vol * avolc
        vol *= np.exp(-0.5 * nu2 * dt + nu * dz1)

        ito = 0.5 * np.power(vols, 2) * dt
        dw = rho * dz1 + sqrtmrho2 * dz0
        spot *= np.exp(-ito + vols * dw)

        # Calculate payoff


    mc_prices = 0

    return mc_prices

if __name__ == "__main__":
    EXPIRIES = [0.25, 1, 5]
    STRIKES = [[95, 100, 105], [95, 100, 105], [95, 100, 105]]
    ARE_CALLS = [[False, False, False], [False, False, False], [False, False, False]]
    FWD = 100
    PARAMETERS = [0, 1, 2, -0.25]
    NUM_MC = 4
    STEPS_PER_YEAR = 5
    SHIFT = 0.03
    p = prices(EXPIRIES, STRIKES, ARE_CALLS, FWD, SHIFT, PARAMETERS, NUM_MC, STEPS_PER_YEAR)
    print(p)
