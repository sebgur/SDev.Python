""" Calibrate a parametric implied vol model to market data. View diagnostics. """
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sdevpy.maths.metrics import rmse
from sdevpy.market import provider as mdp
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider
from sdevpy.volatility.impliedvol import impliedvol_factory
from sdevpy.volatility.impliedvol.impliedvol_calib import TsIvCalibrator
from sdevpy.volatility.impliedvol import impliedvol
from sdevpy.utilities import timegrids
from sdevpy import logger
logger.configure(sdevpy_level='info')


# Choose underlying and date
name, valdate = "ABC", dt.datetime(2025, 12, 15)

# Get MarketDataProvider
md_prov = MarketDataFileProvider()
cal_prov = CalibrationDataFileProvider()

# Choose model
model_name = 'LogMix3' # TsSvi1, TsSvi2, LogMix2, LogMix3
dump_to_file = True

# Retrieve forward curve
fwd_curve = mdp.get_eq_forward_curves([name], valdate, md_prov)[0]

# Retrieve option data
option_data = md_prov.get_eq_vol_data(name, valdate)
mkt_data = {'option_data': option_data, 'forward_curve': fwd_curve}

# Access data in object
expiries = option_data.expiries
fwds = fwd_curve.value(expiries)
mkt_strikes = option_data.get_strikes(fwd_curve=fwd_curve, to_type='absolute')
mkt_vols = option_data.vols

# Quick check of size consistency
print(f"Number of expiries: {len(expiries)}")
print(f"Number of forwards: {len(fwds)}")
print(f"Number of strike sections: {len(mkt_strikes)}")
print(f"Number of vol sections: {len(mkt_vols)}")
for i in range(len(expiries)):
    print(f"Expiry {i+1} number of strikes/vols: {len(mkt_strikes[i])}/{len(mkt_vols[i])}")

option_data.pretty_print()

# Calibrate
iv_surface = impliedvol_factory.get_new_model(model_name)
calibrator = TsIvCalibrator(iv_surface, {'optimizer': 'SLSQP', 'tol': 1e-6})
calibrator.calibrate(mkt_data)

# Display accuracy of the fit. Estimate model on points and calculate RMSE, plot comparison.
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        exp_idx = n_cols * i + j
        expiry = timegrids.model_time(valdate, expiries[exp_idx])
        fwd = fwds[exp_idx]
        exp_strikes = mkt_strikes[exp_idx]
        min_k, max_k = exp_strikes[0], exp_strikes[-1]
        m_strikes = np.linspace(0.8 * min_k, 1.2 * max_k, 100)
        m_vols = iv_surface.black_volatility(expiry, m_strikes, fwd)
        ax.scatter(exp_strikes, mkt_vols[exp_idx], label="market", color='black')
        ax.plot(m_strikes, m_vols, label="model", color='green')
        model_vols = iv_surface.black_volatility(expiry, exp_strikes, fwd)
        vol_rmse = rmse(mkt_vols[exp_idx], model_vols)
        ax.set_title(f"T:{expiry:.2f}, RMSE(bps): {10000.0 * vol_rmse:,.2f}")
        ax.set_xlabel('strike')
        ax.set_ylabel('vol')
        ax.legend()

fig.suptitle('Option vols, Model vs Market', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Dump to file
if dump_to_file:
    file = cal_prov.impliedvol_data_file(name, valdate, model_name)
    iv_surface.dump(file)
    print(f"Dumping model to file: {file}")
