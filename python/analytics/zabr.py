""" Monte-Carlo simulation for ZABR model (vanillas). The model is defined as
    dF = alpha x sigma x F^beta dW
    dsigma = nu x sigma^gamma dZ
    with sigma(0) = 1.0 and <dW, dZ> = rho. SABR is the gamma = 1 case. """
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import analytics.mcsabr as mcsabr
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
    print(f'alpha: {alpha:0.5f}')
    print(f'beta: {beta:.2f}')
    sqrtmrho2 = np.sqrt(1.0 - rho**2)
    vol_floor = 0.0001

    print(f'Gamma: {gamma:.2f}')

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
        # vols = vol

        # Evolve vol
        # nus = nu
        # nus = nu * vol**(gamma - 1.0)
        nus = nu * np.minimum(vol**(gamma - 1.0), 50000.0 * vol_floor**(gamma - 1.0))
        vol *= np.exp(-0.5 * np.power(nus, 2) * dt + nus * dz1)
        # vol = vol + nu * np.power(np.abs(vol), gamma) * dz1
        # vol = np.maximum(vol, 0.0)

        # Evolve spot
        dw = rho * dz1 + sqrtmrho2 * dz0
        ito = 0.5 * np.power(vols, 2) * dt
        spot *= np.exp(-ito + vols * dw)
        # spot = spot + vol * alpha * np.maximum(spot, 0.0001)**beta * dw
        # spot = np.maximum(spot, 0.0)

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
    EXPIRIES = [10.0]#[0.5, 1.0, 5.0, 10.0]
    NSTRIKES = 50
    FWD = 0.0325
    SHIFT = 0.00
    SFWD = FWD + SHIFT
    IS_CALL = False
    ARE_CALLS = [IS_CALL] * NSTRIKES
    ARE_CALLS = [ARE_CALLS] * len(EXPIRIES)
    beta = 0.7
    LNVOL = 0.0873 / FWD**(1.0 - 0.7)
    # Distribution method
    np_expiries = np.asarray(EXPIRIES).reshape(-1, 1)
    PERCENT = np.linspace(0.01, 0.999, NSTRIKES)
    PERCENT = np.asarray([PERCENT] * len(EXPIRIES))
    ITO = -0.5 * LNVOL**2 * np_expiries
    DIFF = LNVOL * np.sqrt(np_expiries) * sp.norm.ppf(PERCENT)
    SSTRIKES = SFWD * np.exp(ITO + DIFF)
    STRIKES = SSTRIKES - SHIFT
    XAXIS = STRIKES

    # PARAMS1 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 0.0}
    # PARAMS2 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 0.5}
    # PARAMS3 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.0}
    # PARAMS4 = {'LnVol': LNVOL, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.5}
    PARAMS5 = {'LnVol': LNVOL, 'Beta': beta, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.7}
    NUM_MC = 500 * 1000
    POINTS_PER_YEAR = 25

    # Calculate MC prices
    mc_timer = timer.Stopwatch("MC")
    mc_timer.trigger()
    # MC_PRICES1 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS1, NUM_MC, POINTS_PER_YEAR)
    # MC_PRICES2 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS2, NUM_MC, POINTS_PER_YEAR)
    # MC_PRICES3 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS3, NUM_MC, POINTS_PER_YEAR)
    # MC_PRICES4 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS4, NUM_MC, POINTS_PER_YEAR)
    # MC_PRICES5 = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS5, NUM_MC, POINTS_PER_YEAR)
    SB_PRICES = mcsabr.price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMS5, NUM_MC, POINTS_PER_YEAR)
    mc_timer.stop()
    mc_timer.print()

    # Convert to IV and compare against approximate closed-form
    import black
    import bachelier
    mc_ivs1 = []
    mc_ivs2 = []
    mc_ivs3 = []
    mc_ivs4 = []
    mc_ivs5 = []
    sb_ivs = []
    for a, expiry in enumerate(EXPIRIES):
        mc_iv1 = []
        mc_iv2 = []
        mc_iv3 = []
        mc_iv4 = []
        mc_iv5 = []
        sb_iv = []
        for j, sstrike in enumerate(SSTRIKES[a]):
            # mc_iv1.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES1[a, j]))
            # mc_iv2.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES2[a, j]))
            # mc_iv3.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES3[a, j]))
            # mc_iv4.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES4[a, j]))
            # mc_iv5.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, MC_PRICES5[a, j]))
            sb_iv.append(black.implied_vol(expiry, SSTRIKES[a, j], IS_CALL, SFWD, SB_PRICES[a, j]))
        mc_ivs1.append(mc_iv1)
        mc_ivs2.append(mc_iv2)
        mc_ivs3.append(mc_iv3)
        mc_ivs4.append(mc_iv4)
        mc_ivs5.append(mc_iv5)
        sb_ivs.append(sb_iv)

    # label1 = f'gam={PARAMS1["Gamma"]}'
    # label2 = f'gam={PARAMS2["Gamma"]}'
    # label3 = f'gam={PARAMS3["Gamma"]}'
    # label4 = f'gam={PARAMS4["Gamma"]}'
    label5 = f'gam={PARAMS5["Gamma"]}'

    xport = np.column_stack((STRIKES[0, :], sb_ivs[0]))
    # xport = np.column_stack((STRIKES[3, :], mc_ivs1[3], mc_ivs2[3], mc_ivs3[3], mc_ivs4[3], mc_ivs5[3], sb_ivs[3]))
    from tools import clipboard
    clipboard.export2d(xport)


    # def plot_zabr(idx):
    #     plt.subplot(2, 2, idx + 1)
        # plt.plot(XAXIS[idx], mc_ivs1[idx], label=label1, color='blue')
        # plt.plot(XAXIS[idx], mc_ivs2[idx], label=label2, color='red')
        # plt.plot(XAXIS[idx], mc_ivs3[idx], label=label3, color='green')
        # plt.plot(XAXIS[idx], mc_ivs4[idx], label=label4, color='purple')
        # plt.plot(XAXIS[idx], mc_ivs5[idx], label=label5, color='cyan')
        # plt.plot(XAXIS[idx], sb_ivs[idx], label="SABR", color='black')
        # plt.xticks(plt.xticks()[0],[f'{x*100:,.1f}%' for x in plt.xticks()[0]])
        # plt.yticks(plt.yticks()[0],[f'{x*100:,.0f}%' for x in plt.yticks()[0]])
        # plt.legend(loc='best')
        # plt.title(f"Expiry: {EXPIRIES[idx]}")

    # plt.figure(figsize=(12, 8))
    # plt.subplots_adjust(hspace=0.40)
    # plot_zabr(0)
    # plot_zabr(1)
    # plot_zabr(2)
    # plot_zabr(3)
    # plt.show()
