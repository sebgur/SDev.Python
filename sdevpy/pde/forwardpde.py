import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.maths import tridiag
from sdevpy.analytics import black
from sdevpy.maths import metrics

########## ToDo (basic) ###################################
# * Analytical expression as additional scheme for early times. This is not really
#   instead of the mollifier. Need to make good decision for where in time
#   the PDE/roll-forward starts, basically. 1D is an obvious candidate. We could
#   specify the 'PDE's min time point" as that point where the analytical expression
#   is used, and throw an error if a time lower than that is requested. No need to
#   throw an error in fact, we could still do it just in case someone asks before
#   that time. The use of the analytical should therefore be optional, to allow possibly someone to refuse
#   to use it in order to go to earlier time steps.
# * Rannacher for early times
# * mass rescaling to 1, forward shifting
# * Convergence tests along time and spot for different schemes and possibilities above

########## ToDo (calibration) ###################################
# * During the warmup in the time direction, we can allocate
#   on each time slice a local vol functional form that is only a function
#   of the spot. This would be a generalized version of the storage
#   of the time interpolation indices for an interpolated surface.
# * A spot parametric local vol would be spot-parametric on predefined
#   time slices, and would for instance take the same parametric form
#   over forward time intervals.
# * We could resolve forward by taking the previous parametric form as
#   starting point.
# * We could use SVI as base and if not enough point, fit only to reduced
#   set of free parameters, the other ones defaulting to good solver starting points.
# * For the (backward) pricing PDE, also allow the standard case of a fully interpolated
#   matrix, using cubic splines with flat extrapolation on both ends and arriving flat
#   first derivative.
# * The choice of time grid during calibration may be guided by the location of the
#   calibration dates. We could decide a certain number of time steps being between
#   each market date, with a certain minimum number especially on the first interval.
# * To check the quality of the calibration, start by comparing against same forward
#   PDE as used in calibration, and then check against backward PDE.


def forward_pde_density(maturity, local_vol, time_config, spot_config, scheme):
    # Build time grid
    t_grid = build_timegrid(maturity, time_config)
    n_timegrid = t_grid.shape[0]

    # Build spot grid
    x, dx, spot_idx = build_spotgrid(maturity, spot_config)
    scheme['dx'] = dx
    scheme['spot_idx'] = spot_idx

    # Initial probability
    p = mollifier(x, 0.0, scheme['dx'], scheme['mollifier'])

    # Backward reduction
    for i in range(n_timegrid - 1):
        ts = t_grid[i]
        te = t_grid[i + 1]

        # Roll forward from ts to te
        p = roll_forward(p, x, ts, te, local_vol, scheme)

        # Check sum at te
        mass = np.trapezoid(p, x)
        # print(f"Mass: {mass:.6f}")

    return x, p


def build_timegrid(maturity, config):
    n_timesteps = config['n_steps']
    return np.linspace(0.0, maturity, n_timesteps)


def build_spotgrid(maturity, config):
    mesh_percentile = config['percentile']
    mesh_vol = config['mesh_vol']
    n_meshes = config['n_meshes']
    p = norm.ppf(1.0 - mesh_percentile)
    x_max = mesh_vol * np.sqrt(maturity) * p
    n_half = int(n_meshes / 2)
    dx = x_max / n_half
    x_grid = np.zeros(2 * n_half + 1)
    x = 0.0
    for i in range(n_half):
        x = x + dx
        x_grid[n_half + 1 + i] = x
        x_grid[n_half - 1 - i] = -x

    return x_grid, dx, n_half


def mollifier(x, x0, dx, k=1.5):
    eps = (k * dx)**2
    p = np.exp(-0.5 * (x - x0)**2 / eps) / np.sqrt(2.0 * np.pi * eps)
    # p /= np.trapezoid(p, x)
    return p


def roll_forward(p, x, ts, te, local_vol, scheme):
    """ Roll the density forward from time ts to te (ts < te) """
    theta = scheme['theta']
    dx = scheme['dx'] # Assuming homogeneous x grid
    one_m_theta = 1.0 - theta
    n_x = x.shape[0]
    dt = te - ts
    a = 1.0 / dx**2 + 0.5 / dx
    b = 2.0 / dx**2
    c = 1.0 / dx**2 - 0.5 / dx

    # Calculate result vector using previous probabilities
    lv = local_vol(ts, x)
    one_m_theta_dt_2 = one_m_theta * dt / 2.0
    y = np.zeros(n_x)
    for j in range(n_x):
        p_tmp = (1.0 - one_m_theta_dt_2 * b * lv[j]**2) * p[j]

        if j < n_x - 1: # Beyond that the probability is 0
            p_tmp += one_m_theta_dt_2 * a * lv[j + 1]**2 * p[j + 1]

        if j > 0: # Before that the probability is 0
            p_tmp += one_m_theta_dt_2 * c * lv[j - 1]**2 * p[j - 1]

        y[j] = p_tmp

    # Calculate band vectors for tridiagonal system
    lv = local_vol(te, x)
    theta_dt_2 = theta * dt / 2.0
    upper = np.zeros(n_x - 1)
    main = np.zeros(n_x)
    lower = np.zeros(n_x - 1)
    for j in range(n_x):
        main[j] = (1.0 + theta_dt_2 * b * lv[j]**2)

        if j < n_x - 1:
            upper[j] = -theta_dt_2 * a * lv[j + 1]**2

        if j > 0:
            lower[j - 1] = -theta_dt_2 * c * lv[j - 1]**2

    # Solve tridiagonal system
    p_new = tridiag.solve(upper, main, lower, y)

    # ToDo: Rescale/recenter here if required?

    return p_new


if __name__ == "__main__":
    maturity = 2.5
    spot = 100.0
    r = 0.0
    q = 0.0
    r_disc = 0.0
    fwd = spot * np.exp((r - q) * maturity)
    atm_vol = 0.20

    def my_lv(t, x_grid):
        """ As a function of log forward moneyness """
        # vol_atm = 0.20
        skew = -0.1
        return np.asarray([atm_vol for x in x_grid])
        # return np.asarray([np.maximum(0.01, atm_vol + skew * x) for x in x_grid])

    #### Diagnostics (single run) #################################################################
    time_config = {'n_steps': 500}
    spot_config = {'n_meshes': 200, 'mesh_vol': atm_vol, 'percentile': 1e-6}
    scheme = {'theta': 1.0, 'mollifier': 1.5}
    print(f"Time steps: {time_config['n_steps']}")
    print(f"Spot steps: {spot_config['n_meshes']}")

    # Calculate probability density
    x, p = forward_pde_density(maturity, my_lv, time_config, spot_config, scheme)

    ## Check density ##
    # Range
    n_dev = 4
    stdev = atm_vol * np.sqrt(maturity)
    x_max = stdev * n_dev

    # PDE
    pde_x = []
    pde_p = []
    for u, v in zip(x, p):
        if np.abs(u) < x_max:
            pde_x.append(u)
            pde_p.append(v)

    # Closed-form
    cf_x = np.linspace(-x_max, x_max, 100)
    cf_p = norm.pdf(cf_x, loc=-0.5 * stdev**2, scale=stdev)

    # Calculate diffs
    cf_all = norm.pdf(pde_x, loc=-0.5 * stdev**2, scale=stdev)
    diff = metrics.rmse(pde_p, cf_all)

    print(f"Int(cf): {np.trapezoid(cf_p, cf_x)}")
    print(f"Int(pde): {np.trapezoid(pde_p, pde_x)}")
    print(f"Diff: {diff:.6f}")

    # Plot
    plt.plot(pde_x, pde_p, label="PDE", color='red')
    plt.plot(cf_x, cf_p, label="CF", color='blue')
    plt.legend()
    plt.show()

    ## Check option prices ##
    strikes = np.linspace(0.50 * spot, 2.0 * spot, 16)
    is_call = True

    print("Calculate prices with Closed-Form")
    cf_prices = black.price(maturity, strikes, is_call, fwd, atm_vol)
    # Intrinsic values
    it_prices = np.maximum(fwd - strikes, 0.0)

    print("Calculate prices with PDE")
    s = fwd * np.exp(x)
    pde_prices = []
    for k in strikes:
        payoff = np.maximum(s - k, 0.0)
        weighted_payoff = payoff * p
        pde_prices.append(np.trapezoid(weighted_payoff, x))

    for k, c, p in zip(strikes, cf_prices, pde_prices):
        print(f"{k:.0f},{c:.2f},{p:.2f}")

    plt.plot(strikes, pde_prices - it_prices, label="PDE", color='red')
    plt.plot(strikes, cf_prices - it_prices, label="CF", color='blue')
    plt.legend()
    plt.show()

    #### Diagnostics (convergence) ################################################################
