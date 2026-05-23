import datetime as dt
import numpy as np
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.pde.forwardpde import calculate_densities
from sdevpy.volatility.localvol.black_calib import calib_lv_black
from sdevpy.utilities import timegrids
from sdevpy.tests import test_localvol


# Init data
spot, r, q = 100.0, 0.04, 0.01
# atm_vol = 0.20
n_dev = 4 # Distribution display range in number of stdevs
n_rows, n_cols = 3, 2 # n_rows * n_cols must match number of maturities
valdate = dt.datetime(2025, 12, 15)
expiries = [dt.datetime(2026, 1, 15), dt.datetime(2026, 5, 15), dt.datetime(2026, 12, 15),
            dt.datetime(2027, 5, 15), dt.datetime(2027, 12, 15), dt.datetime(2030, 12, 15)]

# Define IV surface and calibrate Black LV
print("Define IV surface")
iv_surface = test_localvol.make_tssvi1()
maturities = timegrids.model_time(valdate, expiries)
calib_fwds = spot * np.exp((r - q) * maturities)
calib_strikes = calib_fwds
lv_calib = calib_lv_black(iv_surface, valdate, expiries, calib_strikes, calib_fwds)
lv = lv_calib['lv']
# print(lv.vol_grid)
# # lv.vol_grid[0] = 0.15
# # lv.vol_grid[1] = 0.15
# # lv.vol_grid[2] = 0.15
# lv.vol_grid[3] += 0.05
# # lv.vol_grid[4] = 0.15
# # lv.vol_grid[5] = 0.15
# print(lv.vol_grid)
# lv.refresh_sections()
calib_vols = lv_calib['calib_vols']
print(f"Calib target vols: {calib_vols}")
print(f"LV time grid: {np.asarray(lv.t_grid)}")

# lv_value = lv.value
# lvs = lv_value(0.5, [-0.5, 0, 0.5])
# print(f"Check used lvs: {lvs}")

# # Local Vol
# def my_lv(t, x_grid):
#     """ As a function of log forward moneyness """
#     return np.asarray([atm_vol for x in x_grid])

# PDE config
print("Define PDE config")
mesh_vol = calib_vols[0]
pde_config = PdeConfig(n_timesteps=50, n_meshes=250, mesh_vol=mesh_vol, scheme='rannacher',
                        rescale_x=True, rescale_p=True, iv_surface=iv_surface)
print(f"Time steps: {pde_config.n_timesteps}")
print(f"Spot steps: {pde_config.n_meshes}")
print(f"Mesh vol: {mesh_vol*100:.2f}%")

start_timer = time.time()

# Run PDE to calculate densities at each maturity
density_reports = calculate_densities(maturities, lv.value, pde_config)
# density_reports = calculate_densities(maturities, my_lv, pde_config)

# Calculate PDE option prices and compare against closed-form
reports = []
for r_idx, density_report in enumerate(density_reports):
    maturity = density_report['end_time']
    x = density_report['x_grid']
    p = density_report['p_grid']
    cf_vol = calib_vols[r_idx]

    ## Check density ##
    stdev = cf_vol * np.sqrt(maturity)
    cf_dens = norm.pdf(x, loc=-0.5 * stdev**2, scale=stdev)
    dens_diff = metrics.rmse(p, cf_dens)

    # For display
    x_max = stdev * n_dev # Display range
    disp_pde_x = []
    disp_pde_p = []
    for u, v in zip(x, p, strict=True):
        if np.abs(u) < x_max:
            disp_pde_x.append(u)
            disp_pde_p.append(v)

    # Closed-form (display)
    disp_cf_x = np.linspace(-x_max, x_max, 100)
    disp_cf_p = norm.pdf(disp_cf_x, loc=-0.5 * stdev**2, scale=stdev)

    ## Check prices ##
    fwd = spot * np.exp((r - q) * maturity)
    strikes = np.linspace(0.50 * fwd, 2.0 * fwd, 16)
    cf_prices = black.price_straddles(maturity, strikes, fwd, cf_vol)

    # Calculate PDE prices
    s = fwd * np.exp(x)
    pde_prices = []
    for k in strikes:
        payoff = np.maximum(s - k, 0.0) + np.maximum(k - s, 0.0)
        weighted_payoff = payoff * p
        pde_prices.append(np.trapezoid(weighted_payoff, x))

    price_diff = metrics.rmse(pde_prices, cf_prices)

    pde_prices, cf_prices = np.asarray(pde_prices), np.asarray(cf_prices)
    report = {'acc(dens)': dens_diff*100.0, 'int(cf)': np.trapezoid(cf_dens, x), 'int(pde)': np.trapezoid(p, x),
              'disp_pde_x': disp_pde_x, 'disp_pde_p': disp_pde_p, 'disp_cf_x': disp_cf_x, 'disp_cf_p': disp_cf_p,
              'strikes': strikes, 'pde_prices': pde_prices, 'cf_prices': cf_prices,
              'acc(price)': price_diff / cf_prices.mean()*100.0, 'maturity': maturity}
    reports.append(report)

# Result
runtime = time.time() - start_timer
for r in reports:
    print(f"Maturity: {r['maturity']}")
    print(f"PDE prices: {r['pde_prices']}")
    print(f"CF prices: {r['cf_prices']}")
    print(f"Accuracy(dens): {r['acc(dens)']:.3f}")
    print(f"Accuracy(price): {r['acc(price)']:.3f}%")

dens_accuracy = np.asarray([report['acc(dens)'] for report in reports]).sum()
price_accuracy = np.asarray([report['acc(price)'] for report in reports]).sum()
print(f"Runtime: {runtime:.2f}s")
print(f"Total accuracy(dens): {dens_accuracy:.3f}")
print(f"Total accuracy(price): {price_accuracy:.3f}%")


#### Plot ####
# Density
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        mty_idx = n_cols * i + j
        r = reports[mty_idx]
        ax = axes[i, j]
        ax.plot(r['disp_pde_x'], r['disp_pde_p'], label="PDE", color='red')
        ax.plot(r['disp_cf_x'], r['disp_cf_p'], label="CF", color='blue')
        ax.set_title(f"T={maturities[mty_idx]:.3f}, accuracy = {r['acc(dens)']:.5f}")
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
        ax.set_title(f"T={maturities[mty_idx]:.3f}, accuracy = {r['acc(price)']:.5f}%")
        ax.set_xlabel('strike')
        ax.set_ylabel('price')
        ax.legend()

fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
