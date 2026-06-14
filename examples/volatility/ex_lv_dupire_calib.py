import datetime as dt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.volatility.localvol import localvol_factory as lvf
from sdevpy.market import provider as mdp
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.volatility.impliedvol import impliedvol_factory
from sdevpy.volatility.localvol.dupire_calib import calib_lv_dupire
from sdevpy.utilities import timegrids
from sdevpy.utilities.timegrids import TimeGridBucket
from sdevpy.utilities import dates as dts
from sdevpy.utilities.algos import upper_bound


# Choose underlying and date
name, valdate, model_name = "ABC", dt.datetime(2025, 12, 15), 'LogMix3'

# Get MarketDataProvider
md_prov = MarketDataFileProvider()

# Choose LV diagnostic grids
test_tenors = ['1M', '3M', '6M', '9M', '1Y', '2Y'] # Must have len = 6
n_test_strikes = 10
lw_p = 0.05 # Low percentile strike
up_p = 1.0 - lw_p # High percentile strike

################ Retrieve input data ##############################################################
# Retrieve pre-calibrated implied vol surface
iv_surface = impliedvol_factory.get_impliedvol(name, valdate, model_name)

# Retrieve forward curve
fwd_curve = mdp.get_eq_forward_curves([name], valdate, md_prov)[0]

# Define expiries at which we will observe the accuracy
test_expiries = [dts.advance(valdate, tenor) for tenor in test_tenors]
test_times =  timegrids.model_time(valdate, test_expiries)

# Calculate observation strikes and closed-form straddle prices/IVs
percentiles = np.linspace(lw_p, up_p, n_test_strikes)
n_quantiles = norm.ppf(percentiles)
fwds, strikes, cf_prices, cf_ivs = [], [], [], []
for expiry, time in zip(test_expiries, test_times, strict=True):
    fwd = fwd_curve.value(expiry)
    atm = iv_surface.black_volatility(time, fwd, fwd)
    stdev = atm * np.sqrt(time)
    exp_strikes = fwd * np.exp(-0.5 * stdev * stdev + stdev * n_quantiles)
    call = iv_surface.forward_price(time, exp_strikes, True, fwd)
    put = iv_surface.forward_price(time, exp_strikes, False, fwd)
    fwds.append(fwd)
    strikes.append(exp_strikes)
    cf_prices.append(call + put)
    cf_ivs.append(iv_surface.black_volatility(time, exp_strikes, fwd))

################ Calibrate LV #####################################################################
# Granularity of the LV matrix
points_per_year = 25
n_strikes = 100

# Define calibration horizon
tmax = test_times.max()

# Define time grid by time buckets
time_buckets = []
time_buckets.append(TimeGridBucket(start=0.0, end=0.1, n_points=20))
time_buckets.append(TimeGridBucket(start=0.1, end=0.5, n_points=40))
time_buckets.append(TimeGridBucket(start=0.5, end=1.0, n_points=25))
time_buckets.append(TimeGridBucket(start=1.0, end=10.0, n_points=100))

# Launch Dupire calibration
lv_calib = calib_lv_dupire(iv_surface, points_per_year=points_per_year, n_strikes=n_strikes,
                           tmax=tmax)#, time_buckets=time_buckets, low_percent=0.01)#, t_grid=lv_calib_times)
lv_t = lv_calib['t_grid']
lv_moneyness = lv_calib['moneyness']
lv_matrix = lv_calib['lv_matrix']
lv = lv_calib['lv']

# Dump LV result to file
out_file = lvf.data_file(name, valdate, 'Matrix')
print(f"Dumping LV result to file: {out_file}")
lv.valdate = lv.snapdate = valdate
lv.dump(out_file)

# View the LV along the strike at several expiries
t_idx = [upper_bound(lv_t, tp) for tp in test_times]
plot_lv_t = lv_t[t_idx]

n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        exp_idx = n_cols * i + j
        expiry = plot_lv_t[exp_idx]
        ax.plot(lv_moneyness[exp_idx], lv_matrix[exp_idx], label="model", color='green')
        ax.set_title(f"T:{expiry:.2f}")
        ax.set_xlabel('strike')
        ax.set_ylabel('local vol')
        ax.legend()

fig.suptitle('Local Vol along moneyness, at various expiries', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# View LV along the expiry at several moneynesses
m_idx = [0, 5, 10, 15, 20, 24]
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        k_idx = n_cols * i + j
        k_idx = m_idx[k_idx]
        lv_ = [lv_matrix[tidx][k_idx] for tidx in range(len(lv_t))]
        ax.plot(lv_t, lv_, label="model", color='green')
        ax.set_title(f"Moneyness:{k_idx}")
        ax.set_xlabel('strike')
        ax.set_ylabel('local vol')
        ax.legend()

fig.suptitle('Local Vol along time, at various moneynesses', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
