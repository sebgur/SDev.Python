import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.pde import pdeschemes

########## ToDo (basic) #######################################################
# * Time batching: give vector of times and batch between times
# * X grid rescaling to happen at first time in batch, this way no rescaling at
#   end of previous batch. Mass rescaling could happen at the end, optionally.
# * Analytical expression as additional scheme for early times. This is not really
#   instead of the mollifier. Need to make good decision for where in time
#   the PDE/roll-forward starts, basically. 1D is an obvious candidate. We could
#   specify the 'PDE's min time point" as that point where the analytical expression
#   is used, and throw an error if a time lower than that is requested. No need to
#   throw an error in fact, we could still do it just in case someone asks before
#   that time. The use of the analytical should therefore be optional, to allow possibly someone to refuse
#   to use it in order to go to earlier time steps.
# * Forward shifting?
# * Decide best scheme

########## ToDo (calibration) #################################################
# * Implement implied vol design, parametric time SVI, piecewise time SVI
# * Use seaborn to represent diffs between IV and LV prices on quoted pillars
# * Add 1d solving to ATM only, to do live and Vega with smile solving less often.
# * For each LV parametric form, record two sets of parameters: those against
#   moneyness and those against strike, to achieve both stickiness.
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

def density_step(old_p, old_x, old_dx, old_t, new_t, local_vol, time_config, spot_config, config):
    # Start without rescaling

    # Starting point
    return 0


def density(maturity, local_vol, time_config, spot_config, config):
    # Build time grid
    t_grid = build_timegrid(maturity, time_config)
    n_timegrid = t_grid.shape[0]

    # Build spot grid
    x, dx, spot_idx = build_spotgrid(maturity, spot_config)
    config['dx'] = dx
    config['spot_idx'] = spot_idx

    # Initial probability
    p = mollifier(x, 0.0, config['dx'], config['mollifier'])

    # Forward reduction
    for i in range(n_timegrid - 1):
        ts = t_grid[i]
        te = t_grid[i + 1]

        # Roll forward from ts to te
        p = roll_forward(p, x, ts, te, local_vol, config)

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


def roll_forward(p, x, ts, te, local_vol, config):
    """ Roll the density forward from time ts to te (ts < te) """
    dx = config['dx'] # Assuming homogeneous x grid
    scheme = pdeschemes.scheme(config, ts)
    scheme.local_vol = local_vol
    p_new = scheme.roll_forward(p, x, ts, te, dx)
    return p_new


if __name__ == "__main__":
    spot = 100.0
    r = 0.04
    q = 0.01
    atm_vol = 0.20
    maturities = np.array([0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    n_rows = 3
    n_cols = 2 # n_rows * n_cols must match number of maturities

    def my_lv(t, x_grid):
        """ As a function of log forward moneyness """
        # vol_atm = 0.20
        skew = -0.1
        return np.asarray([atm_vol for x in x_grid])
        # return np.asarray([np.maximum(0.01, atm_vol + skew * x) for x in x_grid])

    #### Diagnostics #################################################################
    time_config = {'n_steps': 50}
    spot_config = {'n_meshes': 250, 'mesh_vol': atm_vol, 'percentile': 1e-6}
    scheme_config = {'mollifier': 1.5, 'type': 'Rannacher', 'theta': 1.0, 'rannacher_time': 0.1}
    print(f"Time steps: {time_config['n_steps']}")
    print(f"Spot steps: {spot_config['n_meshes']}")

    n_dev = 4 # Distribution display range in number of stdevs

    pde_xs = []
    pde_ps = []
    cf_xs = []
    cf_ps = []
    reports = []
    for maturity in maturities:
        # Calculate probability density
        x, p = density(maturity, my_lv, time_config, spot_config, scheme_config)

        ## Check density ##
        stdev = atm_vol * np.sqrt(maturity)
        x_max = stdev * n_dev # Display range

        # PDE
        pde_x = []
        pde_p = []
        for u, v in zip(x, p):
            if np.abs(u) < x_max:
                pde_x.append(u)
                pde_p.append(v)

        # Closed-form (display)
        cf_x = np.linspace(-x_max, x_max, 100)
        cf_p = norm.pdf(cf_x, loc=-0.5 * stdev**2, scale=stdev)

        # Calculate diffs (ToDo: do on all points x, p)
        cf_all = norm.pdf(pde_x, loc=-0.5 * stdev**2, scale=stdev)
        diff = metrics.rmse(pde_p, cf_all)

        report = {'rmse(dens)': diff, 'int(cf)': np.trapezoid(cf_p, cf_x), 'int(pde)': np.trapezoid(pde_p, pde_x),
                  'pde_x': pde_x, 'pde_p': pde_p, 'cf_x': cf_x, 'cf_p': cf_p}

        ## Check option prices ##
        strikes = np.linspace(0.50 * spot, 2.0 * spot, 16)
        is_call = True
        fwd = spot * np.exp((r - q) * maturity)
        cf_prices = black.price(maturity, strikes, is_call, fwd, atm_vol)
        it_prices = np.maximum(fwd - strikes, 0.0) # Intrinsic values

        s = fwd * np.exp(x)
        pde_prices = []
        for k in strikes:
            payoff = np.maximum(s - k, 0.0)
            weighted_payoff = payoff * p
            pde_prices.append(np.trapezoid(weighted_payoff, x))

        report['strikes'] = strikes
        report['pde_prices'] = pde_prices - it_prices
        report['cf_prices'] = cf_prices - it_prices

        reports.append(report)


    # Plot density
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            mty_idx = n_cols * i + j
            r = reports[mty_idx]
            ax = axes[i, j]
            ax.plot(r['pde_x'], r['pde_p'], label="PDE", color='red')
            ax.plot(r['cf_x'], r['cf_p'], label="CF", color='blue')
            ax.set_title(maturities[mty_idx])
            ax.set_xlabel('log-fwd moneyness)')
            ax.set_ylabel('density')
            ax.legend()

    fig.suptitle('Density, PDE vs CF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot prices
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            mty_idx = n_cols * i + j
            r = reports[mty_idx]
            ax.plot(r['strikes'], r['pde_prices'], label="PDE", color='red')
            ax.plot(r['strikes'], r['cf_prices'], label="CF", color='blue')
            ax.set_title(maturities[mty_idx])
            ax.set_xlabel('strike')
            ax.set_ylabel('price')
            ax.legend()

    fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    #### Diagnostics (convergence) ################################################################
