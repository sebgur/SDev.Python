""" Monte-Carlo simulation for SABR models (vanillas) """
import numpy as np
from tools.timegrids import SimpleTimeGridBuilder


def price(expiries, strikes, are_calls, fwd, parameters, num_mc=10000, points_per_year=10):
    """ Calculate vanilla prices under SABR model by Monte-Carlo simulation"""
    scale = fwd
    if scale < 0.0:
        raise ValueError("Negative forward")

    # Build time grid
    time_grid_builder = SimpleTimeGridBuilder(points_per_year=points_per_year)
    time_grid_builder.add_grid(expiries)
    time_grid = time_grid_builder.complete_grid()
    # print("time grid\n", time_grid)

    # Find payoff times
    is_payoff = np.in1d(time_grid, expiries)
    # print("is_payoff\n", is_payoff)

    # Retrieve parameters
    alpha = parameters[0]
    beta = parameters[1]
    nu = parameters[2]
    nu2 = nu**2
    rho = parameters[3]
    sqrtmrho2 = np.sqrt(1.0 - rho**2)

    # Correlation matrix
    corr = np.ones((2, 2))
    corr[0, 1] = corr[1, 0] = 0.0
    # corr[0, 1] = corr[1, 0] = parameters[3]
    # print("corr\n", corr)

    # Set RNG
    means = np.zeros(2)
    seed = 42
    rng = np.random.RandomState(seed)

    # Initialize paths
    spot = np.ones((num_mc, 1)) * fwd
    # print("spot\n", spot)
    vol = np.ones((num_mc, 1)) * 1.0
    # print("vol\n", vol)

    # Loop over time grid
    ts = te = 0
    payoff_count = 0
    mc_prices = []
    for i, t in enumerate(time_grid):
        # print(f"time step {i}")
        ts = te
        te = t
        dt = te - ts
        sqrt_dt = np.sqrt(dt)
        # print("ts\n", ts)
        # print("te\n", te)
        # print("dt\n", dt)
        # print("sqrt_dt\n", sqrt_dt)

        # Evolve
        dz = rng.multivariate_normal(means, corr, size=num_mc) * sqrt_dt
        # dz0 = dz[:, 0]
        # dz1 = dz[:, 1]
        dz0 = dz[:, 0].reshape(-1, 1)
        dz1 = dz[:, 1].reshape(-1, 1)
        # print("BM\n", dz)
        # print("dz0\n", dz0)
        # print("dz1\n", dz1)

        # Scheme
        # avolc = spot**(beta - 1.0)
        # print(spot)
        avolc = alpha * np.minimum(spot**(beta - 1.0), 500.0 * scale**(beta - 1.0))
        # print("avolc\n", avolc)
        vols = vol * avolc
        # print("vol\n", vol)
        vol *= np.exp(-0.5 * nu2 * dt + nu * dz1)
        # print("vols\n", vols)

        ito = 0.5 * np.power(vols, 2) * dt
        dw = rho * dz1 + sqrtmrho2 * dz0
        spot *= np.exp(-ito + vols * dw)
        # print("spot\n", spot)

        # Calculate payoff
        if is_payoff[i]:
            # print("calculate payoff")
            # print("spot\n", spot)
            # print(spot.shape)

            # print("payoff")
            w = [1.0 if is_call else -1.0 for is_call in are_calls[payoff_count]]
            w = np.asarray(w).reshape(1, -1)
            k = np.asarray(strikes[payoff_count]).reshape(1, -1)
            # print(w)
            # print(w.shape)
            # print(k)
            # print(k.shape)
            payoff = np.maximum(w * (spot - k), 0.0)
            # print(payoff)
            # print(payoff.shape)
            rpayoff = np.mean(payoff, axis=0)
            # print(rpayoff)
            mc_prices.append(rpayoff)
            payoff_count += 1


    return np.asarray(mc_prices)

if __name__ == "__main__":
    EXPIRIES = [1, 2]#, 5]
    STRIKES = np.asarray([[-0.01, 0.0, 0.01], [-0.015, 0.0, 0.015]])
    ARE_CALLS = [[False, False, False], [False, True, False]]
    FWD = -0.01
    PARAMETERS = [0.02, 0.5, 0.50, -0.25]
    NUM_MC = 1000
    POINTS_PER_YEAR = 9
    SHIFT = 0.03
    FWD = FWD + SHIFT
    STRIKES = STRIKES + SHIFT
    p = price(EXPIRIES, STRIKES, ARE_CALLS, FWD, PARAMETERS, NUM_MC, POINTS_PER_YEAR)
    print(p)
