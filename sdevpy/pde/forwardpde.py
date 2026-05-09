import numpy as np
import numpy.typing as npt
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.pde import pdeschemes
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.utilities import timegrids


def density_step(old_p: npt.NDArray[np.float64], old_x: npt.NDArray[np.float64], old_dx: float,
                 t_grid: npt.NDArray[np.float64], local_vol, config: PdeConfig):
    """ Forward PDE evolution along t_grid, with optional rescaling of meshes at
        beginning of the step, optional shifting of the density along its x-axis
        to mach the forward, and optional rescaling of density to integrated to 1
        at end of the step """
    # Rescale spot grid
    if config.rescale_x:
        x, dx, spot_idx = build_spotgrid(t_grid[-1], config)
        p = np.interp(x, old_x, old_p, left=0.0, right=0.0)
    else:
        x, dx, p = old_x, old_dx, old_p

    # Forward reduction
    for i in range(t_grid.shape[0] - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], local_vol, config)

    # Shift to match forward
    if config.shift_forward:
        p = shift_forward(x, p)

    # Rescale density
    if config.rescale_p: # Rescale mass to 1.0 at te
        mass = np.trapezoid(p, x)
        print(f"Mass: {mass:.6f}")
        p /= mass

    return x, dx, p


def density(maturity: float, local_vol, config: PdeConfig):
    """ Simple forward PDE for density, without any rescaling """
    # Build time grid
    t_grid = timegrids.build_timegrid(0.0, maturity, config)
    n_timegrid = t_grid.shape[0]

    # Build spot grid
    x, dx, spot_idx = build_spotgrid(maturity, config)

    # Initial probability
    p = lognormal_density(x, 1.0 / 365.0, config.mesh_vol)

    # Forward reduction
    for i in range(n_timegrid - 1):
        p = roll_forward(p, x, dx, t_grid[i], t_grid[i + 1], local_vol, config)

    return x, dx, p


def build_spotgrid(maturity: float, config: dict):
    """ Built spot grid for PDEs """
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


def lognormal_density(x: npt.NDArray[np.float64], t: float, vol: float) -> npt.NDArray[np.float64]:
    """ lognormal density as mollifier """
    var = vol**2 * t
    p = np.exp(-0.5 * x**2 / var) / np.sqrt(2.0 * np.pi * var)
    p /= np.trapezoid(p, x)
    return p


def roll_forward(p: npt.NDArray[np.float64], x: npt.NDArray[np.float64], dx: float, ts: float, te: float,
                 local_vol, pde_config: PdeConfig) -> npt.NDArray[np.float64]:
    """ Roll the density forward from time ts to te (ts < te) """
    scheme = pdeschemes.scheme(pde_config, ts)
    scheme.local_vol = local_vol
    p_new = scheme.roll_forward(p, x, ts, te, dx)
    return p_new


def shift_forward(x: npt.NDArray[np.float64], p: npt.NDArray[np.float64], tol: float=1e-6) -> npt.NDArray[np.float64]:
    """ Shift the density to match the forward. We are not using this for now. """
    ex = np.exp(x)
    pde_forward_m = np.trapezoid(ex * p, x) # If perfect, would be 1.0
    # print(f"PDE forward moneyness: {pde_forward_m}")
    target_forward_m = 1.0
    if np.abs(pde_forward_m - target_forward_m) > tol: # Only shift if beyond tolerance
        shift = np.log(target_forward_m / pde_forward_m)
        x_shifted = x + shift
        p_shifted = np.interp(x, x_shifted, p, left=0.0, right=0.0)
        p_shifted = np.maximum(p_shifted, 0.0)
        pde_forward_m = np.trapezoid(ex * p_shifted, x) # If perfect, would be 1.0
        # print(f"After shift: {pde_forward_m}")
        return p_shifted
    else:
        return p


def calculate_densities(maturities: npt.NDArray[np.float64], lv, pde_config: PdeConfig):
    """ Calculate densities at specified maturities """
    # Initialize spot grid: first maturity if rescaling on x, last maturity otherwise
    spotgrid_tmax = maturities[0] if pde_config.rescale_x else maturities[-1]
    x, dx, spot_idx = build_spotgrid(spotgrid_tmax, pde_config)

    # Initialize density
    start_time = 1.0 / 365.0 # Make sure no payoff is required before that
    p = lognormal_density(x, start_time, pde_config.mesh_vol)

    # Run PDE batches for each maturity
    reports = []
    for mty_idx in range(maturities.shape[0]):
        ts = start_time if mty_idx == 0 else maturities[mty_idx - 1]
        te = maturities[mty_idx]
        step_grid = timegrids.build_timegrid(ts, te, pde_config)
        x, dx, p = density_step(p, x, dx, step_grid, lv, pde_config)

        report = {'start_time': ts, 'end_time': te, 'x_grid': x, 'p_grid': p, 'dx': dx}
        reports.append(report)

    return reports


if __name__ == "__main__":
    spot, r, q, atm_vol = 100.0, 0.04, 0.01, 0.20
    maturities = np.array([0.1, 0.5, 1.0, 2.5, 5.0, 10.0])
    n_dev = 4 # Distribution display range in number of stdevs
    n_rows, n_cols = 3, 2 # n_rows * n_cols must match number of maturities

    def my_lv(t, x_grid):
        """ As a function of log forward moneyness """
        return np.asarray([atm_vol for x in x_grid])

    #### Diagnostics #################################################################
    pde_config = PdeConfig(n_timesteps=50, n_meshes=250, mesh_vol=atm_vol, scheme='rannacher',
                           rescale_x=True, rescale_p=True)
    print(f"Time steps: {pde_config.n_timesteps}")
    print(f"Spot steps: {pde_config.n_meshes}")

    start_timer = time.time()

    # Run PDE
    density_reports = calculate_densities(maturities, my_lv, pde_config)

    # Diagnostics
    reports = []
    total_diff = 0.0
    for density_report in density_reports:
        maturity = density_report['end_time']
        x = density_report['x_grid']
        p = density_report['p_grid']

        ## Check density ##
        stdev = atm_vol * np.sqrt(maturity)
        x_max = stdev * n_dev # Display range

        # PDE
        pde_x = []
        pde_p = []
        for u, v in zip(x, p, strict=True):
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
