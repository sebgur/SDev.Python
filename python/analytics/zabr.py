""" Monte-Carlo simulation for ZABR model (vanillas). The model is defined as
    dF = alpha x sigma x F^beta dW
    dsigma = nu x sigma^gamma dZ
    with sigma(0) = 1.0 and <dW, dZ> = rho. SABR is the gamma = 1 case. """
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
# from analytics.sabr import calculate_alpha
from tools.timegrids import SimpleTimeGridBuilder
from tools import timer


def price(expiries, strikes, are_calls, fwd, parameters, num_mc=10000, points_per_year=10):
    """ Calculate vanilla prices under ZABR model by Monte-Carlo simulation"""
    scale = fwd
    if scale < 0.0:
        raise ValueError("Negative forward")

    # Temporarily turn off the warnings for division by 0. This is because on certain paths,
    # the spot becomes so close to 0 that Python effectively handles it as 0. This results in
    # a warning when taking a negative power of it. However, this is not an issue as Python
    # correctly finds +infinity and since we use a floor, this case is correctly handled.
    np.seterr(divide='ignore')

    # Build time grid
    time_grid_builder = SimpleTimeGridBuilder(points_per_year=points_per_year)
    time_grid_builder.add_grid(expiries)
    time_grid = time_grid_builder.complete_grid()
    num_factors = 2

    # Find payoff times
    is_payoff = np.in1d(time_grid, expiries)

    # Retrieve parameters
    lnvol = parameters['LnVol']
    beta = parameters['Beta']
    nu = parameters['Nu']
    rho = parameters['Rho']
    gamma = parameters['Gamma']
    alpha = calculate_zabr_alpha(lnvol, fwd, beta)
    sqrtmrho2 = np.sqrt(1.0 - rho**2)
    vol_floor = 0.000001

    # Draw all gaussians
    # gaussians = rand.gaussians(num_steps, num_mc, num_factors, rand_method)

    # Define dimensions
    mean = np.zeros(num_factors)
    corr = np.zeros((num_factors, num_factors))
    for c in range(num_factors):
        corr[c, c] = 1.0

    # Draw for each step
    seed = 42
    rng = np.random.RandomState(seed)

    # Initialize paths
    spot = np.ones((2 * num_mc, 1)) * fwd
    vol = np.ones((2 * num_mc, 1)) * 1.0

    # Loop over time grid
    ts = te = 0
    payoff_count = 0
    mc_prices = []
    for i, t in enumerate(time_grid):
        ts = te
        te = t
        dt = te - ts
        sqrt_dt = np.sqrt(dt)

        # Evolve
        dz = rng.multivariate_normal(mean, corr, size=num_mc) * sqrt_dt
        dz = np.concatenate((dz, -dz), axis=0) # Antithetic paths
        dz0 = dz[:, 0].reshape(-1, 1)
        dz1 = dz[:, 1].reshape(-1, 1)

        # Only using LogEuler scheme
        avolc = alpha * np.minimum(spot**(beta - 1.0), 500.0 * scale**(beta - 1.0))
        vols = vol * avolc

        # Evolve vol
        # vol = vol + nu * np.power(np.abs(vol), gamma) * dz1
        nus = nu * np.minimum(vol**(gamma - 1.0), 500.0 * vol_floor**(gamma - 1.0))
        vol *= np.exp(-0.5 * nus**2 * dt + nus * dz1)

        # Evolve spot
        ito = 0.5 * np.power(vols, 2) * dt
        dw = rho * dz1 + sqrtmrho2 * dz0
        spot *= np.exp(-ito + vols * dw)

        # Calculate payoff
        if is_payoff[i]:
            w = [1.0 if is_call else -1.0 for is_call in are_calls[payoff_count]]
            w = np.asarray(w).reshape(1, -1)
            k = np.asarray(strikes[payoff_count]).reshape(1, -1)
            payoff = np.maximum(w * (spot - k), 0.0)
            rpayoff = np.mean(payoff, axis=0)
            mc_prices.append(rpayoff)
            payoff_count += 1

    np.seterr(divide='warn')

    return np.asarray(mc_prices)


def calculate_zabr_alpha(ln_vol, fwd, beta):
    """ Calculate original parameter alpha with our definition in terms of ln_vol, i.e.
        alpha = ln_vol * fwd ^ (1.0 - beta) """
    return ln_vol * fwd ** (1.0 - beta)


if __name__ == "__main__":
    EXPIRIES = [0.5, 1.0, 5.0, 10.0]
    NSTRIKES = 50
    FWD = 0.04
    SHIFT = 0.00
    SFWD = FWD + SHIFT
    IS_CALL = False
    ARE_CALLS = [IS_CALL] * NSTRIKES
    ARE_CALLS = [ARE_CALLS] * len(EXPIRIES)
    LNVOL = 0.23
    # Distribution method
    np_expiries = np.asarray(EXPIRIES).reshape(-1, 1)
    PERCENT = np.linspace(0.01, 0.99, NSTRIKES)
    PERCENT = np.asarray([PERCENT] * len(EXPIRIES))
    ITO = -0.5 * LNVOL**2 * np_expiries
    DIFF = LNVOL * np.sqrt(np_expiries) * sp.norm.ppf(PERCENT)
    SSTRIKES = SFWD * np.exp(ITO + DIFF)
    STRIKES = SSTRIKES - SHIFT
    XAXIS = STRIKES

    PARAMS00 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 0.0}
    PARAMS05 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 0.5}
    PARAMS10 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.0}
    PARAMS15 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.1}
    PARAMS17 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.2}
    NUM_MC = 100 * 1000
    POINTS_PER_YEAR = 25

    # Calculate MC prices
    mc_timer = timer.Stopwatch("MC")
    mc_timer.trigger()
    MC_PRICES00 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS00, NUM_MC, POINTS_PER_YEAR)
    MC_PRICES05 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS05, NUM_MC, POINTS_PER_YEAR)
    MC_PRICES10 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS10, NUM_MC, POINTS_PER_YEAR)
    MC_PRICES15 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS15, NUM_MC, POINTS_PER_YEAR)
    MC_PRICES17 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS17, NUM_MC, POINTS_PER_YEAR)
    mc_timer.stop()
    mc_timer.print()

    # print(MC_PRICES)

    # Convert to IV and compare against approximate closed-form
    import black
    import bachelier
    mc_ivs00 = []
    mc_ivs05 = []
    mc_ivs10 = []
    mc_ivs15 = []
    mc_ivs17 = []
    for a, expiry in enumerate(EXPIRIES):
        mc_iv00 = []
        mc_iv05 = []
        mc_iv10 = []
        mc_iv15 = []
        mc_iv17 = []
        for j, sstrike in enumerate(SSTRIKES[a]):
            mc_iv00.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES00[a, j]))
            mc_iv05.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES05[a, j]))
            mc_iv10.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES10[a, j]))
            mc_iv15.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES15[a, j]))
            mc_iv17.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES17[a, j]))
            # mc_iv.append(bachelier.implied_vol(expiry, STRIKES[a, j], IS_CALL, FWD, MC_PRICES[a, j]))
            # mc_iv1.append(bachelier.implied_vol(expiry, STRIKES[a, j], IS_CALL, FWD, MC_PRICES1[a, j]))
        mc_ivs00.append(mc_iv00)
        mc_ivs05.append(mc_iv05)
        mc_ivs10.append(mc_iv10)
        mc_ivs15.append(mc_iv15)
        mc_ivs17.append(mc_iv17)

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.40)
    plt.subplot(2, 2, 1)
    plt.plot(XAXIS[0], mc_ivs00[0], label='gam=0', color='blue')
    plt.plot(XAXIS[0], mc_ivs05[0], label='gam=0.5', color='red')
    plt.plot(XAXIS[0], mc_ivs10[0], label='gam=1', color='green')
    plt.plot(XAXIS[0], mc_ivs15[0], label='gam=1.5', color='purple')
    plt.plot(XAXIS[0], mc_ivs17[0], label='gam=1.7', color='cyan')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[0]}")
    plt.subplot(2, 2, 2)
    plt.plot(XAXIS[1], mc_ivs00[1], label='gam=0', color='blue')
    plt.plot(XAXIS[1], mc_ivs05[1], label='gam=0.5', color='red')
    plt.plot(XAXIS[1], mc_ivs10[1], label='gam=1', color='green')
    plt.plot(XAXIS[1], mc_ivs15[1], label='gam=1.5', color='purple')
    plt.plot(XAXIS[1], mc_ivs17[1], label='gam=1.7', color='cyan')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[1]}")
    plt.subplot(2, 2, 3)
    plt.plot(XAXIS[2], mc_ivs00[2], label='gam=0', color='blue')
    plt.plot(XAXIS[2], mc_ivs05[2], label='gam=0.5', color='red')
    plt.plot(XAXIS[2], mc_ivs10[2], label='gam=1', color='green')
    plt.plot(XAXIS[2], mc_ivs15[2], label='gam=1.5', color='purple')
    plt.plot(XAXIS[2], mc_ivs17[2], label='gam=1.7', color='cyan')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[2]}")
    plt.subplot(2, 2, 4)
    plt.plot(XAXIS[3], mc_ivs00[3], label='gam=0', color='blue')
    plt.plot(XAXIS[3], mc_ivs05[3], label='gam=0.5', color='red')
    plt.plot(XAXIS[3], mc_ivs10[3], label='gam=1', color='green')
    plt.plot(XAXIS[3], mc_ivs15[3], label='gam=1.5', color='purple')
    plt.plot(XAXIS[3], mc_ivs17[3], label='gam=1.7', color='cyan')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[3]}")

    plt.show()
