from pathlib import Path
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sdevpy.volatility.localvol import localvol_factory as lvf
from sdevpy.volatility.localvol.lvsection_calib import calibrate_lv_bysections
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.market.eqforward import get_forward_curves
from sdevpy.maths import metrics
from sdevpy import logsetup
logsetup.configure()


verbose, n_digits = False, 6
np.set_printoptions(suppress=True, precision=n_digits)
name = "ABC"
valdate = dt.datetime(2025, 12, 15)
# model_name = 'VSVI'
model_name = 'BiExp'

# Calibration config (COBYLA, SLSQP)
lv_data_folder = lvf.test_data_folder()
config = {'model': model_name, 'store_date': valdate, 'lv_folder': lv_data_folder,
          'pde_timesteps': 50, 'pde_spotsteps': 100,
          'optimizer': 'LeastSquares', 'tol': None, 'maxiter': 100, 'sol_as_init': False,
          'popsize': 5}

# Calibrate LV
print("Launching calibration")
calib_result = calibrate_lv_bysections(valdate, name, config, verbose=True, calc_pde_vols=True)
lv = calib_result['lv']

# Dump LV result to file
out_folder = lvf.test_data_folder()
fname = dt.datetime.now().strftime(dts.DATE_FILE_FORMAT) + "." + config['model']
out_file = Path(out_folder) / name / (fname + ".json")
print(f"Dumping LV result to file: {out_file}")
lv.dump(out_file)

# ################ DIAGNOSTICS ################################################################
# Retrieve results for diagnostics
pde_vols = calib_result['pde_vols']
surface_data = calib_result['iv_data']
expiries = surface_data.expiries
expiry_grid = np.array([timegrids.model_time(valdate, expiry) for expiry in expiries])

# Retrieve forward curve
fwd_curve = get_forward_curves([name], valdate)[0]

# fwds = surface_data.forwards
fwds = fwd_curve.value(expiries)
strike_surface = surface_data.get_strikes(fwd_curve=fwd_curve, to_type='absolute')
vol_surface = surface_data.vols

# Calculate RMSEs on vols
vol_rmses = []
for exp_idx in range(len(expiry_grid)):
    vol_rmses.append(10000.0 * metrics.rmse(vol_surface[exp_idx], pde_vols[exp_idx]))

# Display price results
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        exp_idx = n_cols * i + j
        strikes = strike_surface[exp_idx]
        ax.plot(strikes, pde_vols[exp_idx], label="PDE", color='red')
        ax.plot(strikes, vol_surface[exp_idx], label="CF", color='blue')
        ax.set_title(f"T:{expiry_grid[exp_idx]:.2f}, RMSE: {vol_rmses[exp_idx]:.4f}")
        ax.set_xlabel('strike')
        ax.set_ylabel('price')
        ax.legend()

fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Display LV results
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        exp_idx = n_cols * i + j
        expiry = expiry_grid[exp_idx]
        vol = vol_surface[exp_idx].mean()
        stdev = vol * np.sqrt(expiry)
        print(f"Params at {expiry:.3f}, {lv.params(exp_idx)}")
        xs = np.linspace(-3.0 * stdev, 3.0 * stdev, 100)
        lvs = lv.value(expiry, xs)
        ax.plot(xs, lvs, label="LV", color='blue')
        # strikes = strike_surface[exp_idx]
        # lvs = lv.value(expiry_grid[exp_idx], np.log(strikes / fwds[exp_idx]))
        # ax.plot(strikes, lvs, label="LV", color='blue')
        ax.set_title(f"T:{expiry:.2f}, RMSE: {vol_rmses[exp_idx]:.4f}")
        ax.set_xlabel('strike')
        ax.set_ylabel('price')
        ax.legend()

fig.suptitle('Option prices, PDE vs CF', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
