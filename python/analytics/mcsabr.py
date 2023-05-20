""" Monte-Carlo simulation for SABR models (vanillas) """
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
from analytics.sabr import calculate_alpha
from tools.timegrids import SimpleTimeGridBuilder
from tools import timer


def price(expiries, strikes, are_calls, fwd, parameters, num_mc=10000, points_per_year=10,
          scheme='LogAndersen'):
    """ Calculate vanilla prices under SABR model by Monte-Carlo simulation"""
    scale = fwd
    if scale < 0.0:
        raise ValueError("Negative forward")

    # Temporarily turn of the warnings for division by 0. This is because on certain paths,
    # the spot becomes so close to 0 and Python effectively handles it as 0. This results in
    # a warning when taking a negative power of it. However, this is not an issue as Python
    # correctly finds +infinity and since we use a floor, this case is correctly handled.
    # So we remove the warning temporarily for clarity of outputs.
    np.seterr(divide='ignore')

    # Build time grid
    time_grid_builder = SimpleTimeGridBuilder(points_per_year=points_per_year)
    time_grid_builder.add_grid(expiries)
    time_grid = time_grid_builder.complete_grid()
    print("time grid size: ", len(time_grid))
    num_factors = 2

    # Find payoff times
    is_payoff = np.in1d(time_grid, expiries)

    # Retrieve parameters
    lnvol = parameters['LnVol']
    beta = parameters['Beta']
    nu = parameters['Nu']
    rho = parameters['Rho']
    alpha = calculate_alpha(lnvol, fwd, beta)
    nu2 = nu**2
    sqrtmrho2 = np.sqrt(1.0 - rho**2)

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
        dz = np.concatenate((dz, -dz), axis=0)
        dz0 = dz[:, 0].reshape(-1, 1)
        dz1 = dz[:, 1].reshape(-1, 1)

        # Scheme
        if scheme in ('LogEuler', 'LogAndersen'):
            avolc = alpha * np.minimum(spot**(beta - 1.0), 500.0 * scale**(beta - 1.0))
            vols = vol * avolc
        else:
            vols = vol

        # Evolve vol
        vol *= np.exp(-0.5 * nu2 * dt + nu * dz1)

        # Evolve spot
        if scheme == 'LogEuler':
            ito = 0.5 * np.power(vols, 2) * dt
            dw = rho * dz1 + sqrtmrho2 * dz0
            spot *= np.exp(-ito + vols * dw)
        elif scheme == 'LogAndersen':
            vole = vol * avolc
            intvol2 = 0.5 * (np.power(vols, 2) + np.power(vole, 2)) * dt
            ito = 0.5 * intvol2
            linear1 = (vole - vols) / nu
            linear2 = np.sqrt(intvol2 / dt)
            spot *= np.exp(-ito + linear1 * rho + linear2 * sqrtmrho2 * dz0)
        elif scheme == 'Andersen': # Does not seem to work well
            vole = vol
            spot += alpha * np.abs(spot)**beta * (sqrtmrho2 * vols * dz0 + rho / nu * (vole - vols))
        else:
            raise ValueError("Unknown scheme in MCSABR: " + scheme)

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


if __name__ == "__main__":
    EXPIRIES = [0.05, 0.10, 0.25, 0.5]
    NSTRIKES = 50
    FWD = -0.005
    SHIFT = 0.03
    SFWD = FWD + SHIFT
    IS_CALL = False
    ARE_CALLS = [IS_CALL] * NSTRIKES
    ARE_CALLS = [ARE_CALLS] * len(EXPIRIES)
    LNVOL = 0.25
    # Spread method
    # SPREADS = np.linspace(-200, 200, NSTRIKES)
    # SPREADS = np.asarray([SPREADS] * len(EXPIRIES))
    # STRIKES = FWD + SPREADS / 10000.0
    # SSTRIKES = STRIKES + SHIFT
    # XAXIS = SPREADS
    # Distribution method
    np_expiries = np.asarray(EXPIRIES).reshape(-1, 1)
    PERCENT = np.linspace(0.01, 0.99, NSTRIKES)
    PERCENT = np.asarray([PERCENT] * len(EXPIRIES))
    ITO = -0.5 * LNVOL**2 * np_expiries
    DIFF = LNVOL * np.sqrt(np_expiries) * sp.norm.ppf(PERCENT)
    SSTRIKES = SFWD * np.exp(ITO + DIFF)
    STRIKES = SSTRIKES - SHIFT
    XAXIS = STRIKES

    PARAMETERS = {'LnVol': LNVOL, 'Beta': 0.1, 'Nu': 0.50, 'Rho': -0.25}
    NUM_MC = 1000 * 1000
    POINTS_PER_YEAR = 200
    SCHEME = 'LogAndersen'
    # SCHEME = 'LogEuler'

    # Calculate MC prices
    mc_timer = timer.Stopwatch("MC")
    mc_timer.trigger()
    MC_PRICES = price(EXPIRIES, SSTRIKES, ARE_CALLS, SFWD, PARAMETERS, NUM_MC, POINTS_PER_YEAR,
                      scheme=SCHEME)
    mc_timer.stop()
    mc_timer.print()

    # Convert to IV and compare against approximate closed-form
    import black
    import bachelier
    import sabr
    mc_ivs = []
    cf_ivs = []
    n_ivs = []
    for a, expiry in enumerate(EXPIRIES):
        mc_iv = []
        cf_iv = []
        n_iv = []
        for j, sstrike in enumerate(SSTRIKES[a]):
            mc_iv.append(black.implied_vol(expiry, sstrike, IS_CALL, SFWD, MC_PRICES[a, j]))
            cf_iv.append(sabr.implied_vol_vec(expiry, sstrike, SFWD, PARAMETERS))
            n_iv.append(bachelier.implied_vol_solve(expiry, STRIKES[a, j], IS_CALL, FWD,
                                                    MC_PRICES[a, j]))
        mc_ivs.append(mc_iv)
        cf_ivs.append(cf_iv)
        n_ivs.append(n_iv)

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.40)
    plt.subplot(2, 2, 1)
    plt.plot(XAXIS[0], mc_ivs[0], label='MC')
    plt.plot(XAXIS[0], cf_ivs[0], label='CF')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[0]}")
    plt.subplot(2, 2, 2)
    plt.plot(XAXIS[1], mc_ivs[1], label='MC')
    plt.plot(XAXIS[1], cf_ivs[1], label='CF')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[1]}")
    plt.subplot(2, 2, 3)
    plt.plot(XAXIS[2], mc_ivs[2], label='MC')
    plt.plot(XAXIS[2], cf_ivs[2], label='CF')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[2]}")
    plt.subplot(2, 2, 4)
    plt.plot(XAXIS[3], mc_ivs[3], label='MC')
    plt.plot(XAXIS[3], cf_ivs[3], label='CF')
    plt.legend(loc='best')
    plt.title(f"Expiry: {EXPIRIES[3]}")

    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.40)
    plt.subplot(2, 2, 1)
    plt.plot(XAXIS[0], n_ivs[0], label='MC')
    plt.legend(loc='best')
    plt.title(f"NVOL Expiry: {EXPIRIES[0]}")
    plt.subplot(2, 2, 2)
    plt.plot(XAXIS[1], n_ivs[1], label='MC')
    plt.legend(loc='best')
    plt.title(f"NVOL Expiry: {EXPIRIES[1]}")
    plt.subplot(2, 2, 3)
    plt.plot(XAXIS[2], n_ivs[2], label='MC')
    plt.legend(loc='best')
    plt.title(f"NVOL Expiry: {EXPIRIES[2]}")
    plt.subplot(2, 2, 4)
    plt.plot(XAXIS[3], n_ivs[3], label='MC')
    plt.legend(loc='best')
    plt.title(f"NVOL Expiry: {EXPIRIES[3]}")

    plt.show()
