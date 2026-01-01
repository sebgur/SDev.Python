import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.pde import pdeschemes


def density_step(old_p, old_x, old_dx, t_grid, local_vol, config):
    """ Forward PDE evolution along t_grid, with optional rescaling of meshes at
        beginning of the step and optional rescaling of density at end of the step """
    # Rescale inputs
    if config.rescale_x:
        x, dx, spot_idx = build_spotgrid(t_grid[-1], config)
        p = np.interp(x, old_x, old_p, left=0.0, right=0.0)
    else:
        x, dx, p = old_x, old_dx, old_p

    # Forward reduction
    for i in range(t_grid.shape[0] - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], local_vol, config)

    # Rescale density
    if config.rescale_p: # Rescale mass to 1.0 at te
        mass = np.trapezoid(p, x)
        p /= mass
        # print(f"Mass: {mass:.6f}")

    return x, dx, p


def density(maturity, local_vol, config):
    """ Simple forward PDE for density, without step rescaling of meshes until maturity """
    # Build time grid
    t_grid = build_timegrid(0.0, maturity, config)
    n_timegrid = t_grid.shape[0]

    # Build spot grid
    x, dx, spot_idx = build_spotgrid(maturity, config)

    # Initial probability
    p = lognormal_density(x, 1.0 / 365.0, config.mesh_vol)

    # Forward reduction
    for i in range(n_timegrid - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], local_vol, config)

    # Rescale density
    if config.rescale_p: # Rescale mass to 1.0 at te
        mass = np.trapezoid(p, x)
        p /= mass
        # print(f"Mass: {mass:.6f}")

    return x, dx, p


def build_timegrid(t_start, t_end, config):
    n_steps = config.n_time_steps
    return np.linspace(t_start, t_end, n_steps)


def build_spotgrid(maturity, config):
    mesh_percentile = config.percentile
    mesh_vol = config.mesh_vol
    n_meshes = config.n_meshes
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


def lognormal_density(x, t, vol):
    """ lognormal density to use as mollifier """
    var = vol**2 * t
    p = np.exp(-0.5 * x**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapezoid(p, x)
    return p


def roll_forward(p, x, dx, ts, te, local_vol, pde_config):
    """ Roll the density forward from time ts to te (ts < te) """
    scheme = pdeschemes.scheme(pde_config, ts)
    # print(f"{ts:.4f}-{type(scheme)}")
    scheme.local_vol = local_vol
    p_new = scheme.roll_forward(p, x, ts, te, dx)
    return p_new

def shift_to_match_forward(x, p, target_forward):
    """ Shift the density to match the forward. We are not using this for now.
        This code was written by Claude and has not been tested. """
    ex = np.exp(x)
    cur_forward = np.trapz(ex * p, x)
    if cur_forward <= 0:
        p = np.maximum(p, 0)
        p = p / np.trapz(p, x)
        return p
    alpha = np.log(target_forward / cur_forward)
    x_shifted = x - alpha
    p_shifted = np.interp(x, x_shifted, p, left=0.0, right=0.0)
    p_shifted = np.maximum(p_shifted, 0.0)
    mass = np.trapz(p_shifted, x)
    if mass <= 0:
        p_shifted = np.exp(-0.5 * ((x - np.log(S0)) / (0.1))**2)
        mass = np.trapz(p_shifted, x)
    p_shifted /= mass
    return p_shifted


class PdeConfig:
    def __init__(self, **kwargs):
        self.n_time_steps = kwargs.get('n_time_steps', 25)
        self.n_meshes = kwargs.get('n_meshes', 100)
        self.mesh_vol = kwargs.get('mesh_vol', 0.20)
        self.percentile = kwargs.get('percentile', 1e-6)
        self.mollifier = kwargs.get('mollifier', 1.5)
        self.scheme = kwargs.get('scheme', 'Implicit')
        self.theta = kwargs.get('theta', 0.5)
        self.rannacher_time = kwargs.get('rannacher_time', 7.0 / 365.0)
        self.rescale_x = kwargs.get('rescale_x', True)
        self.rescale_p = kwargs.get('rescale_p', True)


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
    pde_config = PdeConfig(n_time_steps=50, n_meshes=250, mesh_vol=atm_vol, scheme='rannacher',
                           rescale_x=True, rescale_p=True)
    print(f"Time steps: {pde_config.n_time_steps}")
    print(f"Spot steps: {pde_config.n_meshes}")

    n_dev = 4 # Distribution display range in number of stdevs
    use_batches = True

    pde_xs = []
    pde_ps = []
    cf_xs = []
    cf_ps = []
    reports = []

    # Build spot grid (fixed throughout for now)
    if use_batches and pde_config.rescale_x:
        x, dx, spot_idx = build_spotgrid(maturities[0], pde_config)
    else:
        x, dx, spot_idx = build_spotgrid(maturities[-1], pde_config)

    # Start-up density
    start_time = 1.0 / 365.0 # Make sure no payoff is required before that
    p = lognormal_density(x, start_time, pde_config.mesh_vol)

    start_timer = time.time()
    total_diff = 0.0
    for mty_idx in range(maturities.shape[0]):
        maturity = maturities[mty_idx]
        if use_batches:
            ts = start_time if mty_idx == 0 else maturities[mty_idx - 1]
            te = maturities[mty_idx]
            step_grid = build_timegrid(ts, te, pde_config)
            x, dx, p = density_step(p, x, dx, step_grid, my_lv, pde_config)
        else:
            x, dx, p = density(maturity, my_lv, pde_config)

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
        total_diff += diff

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

    # Result
    runtime = time.time() - start_timer
    print(f"Runtime: {runtime:.2f}s")
    print(f"Accuracy: {total_diff*100:.3f}")

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
