import numpy as np


def mc_simulation(num_underlyings, path_simulator, product, disc_rate, spot, vol, num_mc, rng):
    mc_spot = np.ones(shape=(num_mc, num_underlyings)) * spot
    mc_vol = np.ones(shape=(num_mc, num_underlyings)) * vol
    print("Simulating spot paths...")
    mc_future_spots = path_simulator.build_paths(mc_spot, mc_vol, rng)
    print("Discounting payoffs...")
    mc_disc_payoffs = product.disc_payoff(mc_future_spots, disc_rate)
    mc_pv = 0
    for i in range(num_mc):
        mc_pv += mc_disc_payoffs[i][0]
    mc_pv /= num_mc

    mc_delta, mc_gamma, mc_vega = None, None, None

    return mc_pv, mc_delta, mc_gamma, mc_vega
