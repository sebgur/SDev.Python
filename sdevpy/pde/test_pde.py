import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.maths import tridiag

########## ToDo ###################################
# * Would make sense to wrap the PDE resolution forward into function that goes forward
#   from a given ts to a given te, given the vector of probabilities calculated at ts.
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

def local_vol(t, x_grid):
    """ As a function of log forward moneyness """
    vol_atm = 0.20
    skew = -0.1
    return np.asarray([vol_atm for x in x_grid])
    # return np.asarray([np.maximum(0.01, vol_atm + skew * x) for x in x_grid])


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


if __name__ == "__main__":
    maturity = 1.5
    time_config = {'n_steps': 2}
    spot_config = {'n_meshes': 5, 'mesh_vol': 0.20, 'percentile': 1e-4}
    scheme = {'theta': 1.0}

    ### Build time grid ####
    t_grid = build_timegrid(maturity, time_config)
    # print(f"Time grid:\n {t_grid}")
    n_timegrid = t_grid.shape[0]

    #### Build spot grid ####
    x_grid, dx, spot_idx = build_spotgrid(maturity, spot_config)
    # print(f"Spot grid:\n {x_grid}")
    # print(f"dx: {dx}")

    #### Initial probability ####
    # Naive approach (Dirac)
    n_spotgrid = x_grid.shape[0]
    p = np.zeros(n_spotgrid)
    p[spot_idx] = 1.0

    #### Backward reduction ####
    a = 1.0 / dx**2 + 0.5 / dx
    b = 2.0 / dx**2
    c = 1.0 / dx**2 - 0.5 / dx
    for i in range(n_timegrid - 1):
        ts = t_grid[i]
        te = t_grid[i + 1]
        dt = te - ts
        theta = scheme['theta'] # Later this might depend on time (Rannacher)
        one_m_theta = 1.0 - theta
        print(f"\nNew time: {te}")

        ## Calculate result vector using previous probabilities ##
        # Calculate local vol vector
        lv = local_vol(ts, x_grid)
        print(f"Calculated LV at {ts}: {lv}")

        # Calculate result vector
        one_m_theta_dt_2 = one_m_theta * dt / 2.0
        y = np.zeros(n_spotgrid)
        for j in range(n_spotgrid):
            p_tmp = (1.0 - one_m_theta_dt_2 * b * lv[j]**2) * p[j]
            # if j == spot_idx:
            #     print(p_tmp)

            if j < n_spotgrid - 1: # Beyond that the probability is 0
                p_tmp += one_m_theta_dt_2 * a * lv[j + 1]**2 * p[j + 1]

            # if j == spot_idx:
            #     print(p_tmp)

            if j > 0: # Before that the probability is 0
                p_tmp += one_m_theta_dt_2 * c * lv[j - 1]**2 * p[j - 1]

            # if j == spot_idx:
            #     print(p_tmp)


            y[j] = p_tmp

        print(f"Calculated y: {y}")

        ## Calculate band vectors for tridiagonal system ##
        # Calculate local vol vector
        lv = local_vol(te, x_grid)
        print(f"Calculated LV at {te}: {lv}")
        # Calculate bands
        theta_dt_2 = theta * dt / 2.0
        upper = np.zeros(n_spotgrid - 1)
        main = np.zeros(n_spotgrid)
        lower = np.zeros(n_spotgrid - 1)
        for j in range(n_spotgrid):
            main[j] = (1.0 + theta_dt_2 * b * lv[j]**2)

            if j < n_spotgrid - 1:
                upper[j] = -theta_dt_2 * a * lv[j + 1]**2

            if j > 0:
                lower[j - 1] = -theta_dt_2 * c * lv[j - 1]**2

        # Solve tridiagonal system
        x = tridiag.solve(upper, main, lower, y)
        print(f"Calculated x: {x}")
        p = x.copy()

    #### Display ####
    # # PDE
    # pde_x = []
    # pde_p = []
    # for u, v in zip(x_grid, x):
    #     if u > -0.25 and u < 0.25:
    #         pde_x.append(u)
    #         pde_p.append(v)

    # plt.plot(pde_x, pde_p, label="PDE", color='red')

    # # Closed-form at ATM
    # atm_vol = 0.20
    # percentile = 1e-4
    # p = norm.ppf(1.0 - percentile)
    # x_max = atm_vol * np.sqrt(maturity) * p
    # cf_x = np.linspace(-x_max, x_max, 100)
    # cf_p = norm.pdf(cf_x, loc=0.0, scale=atm_vol * np.sqrt(maturity))
    # plt.plot(cf_x, cf_p, label="CF", color='blue')
    # plt.legend()

    # plt.show()