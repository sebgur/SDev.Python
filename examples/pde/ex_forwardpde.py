import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.pde.forwardpde import calculate_densities


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
