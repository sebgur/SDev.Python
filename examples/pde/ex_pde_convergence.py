import datetime as dt
import numpy as np
from scipy.stats import norm
from sdevpy.market import provider as mdp
from sdevpy.volatility.localvol import localvol_factory as lvf
from sdevpy.volatility.localvol.localvol import ConstantLocalVol
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.pde import forwardpde as fpde
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.maths import metrics
from sdevpy.analytics import black


# Specify underlying and date
name, valdate = "ABC", dt.datetime(2025, 12, 15)

# Specify products
tenor = '1y'
expiry = dts.advance(valdate, tenor)
expiry_time = timegrids.model_time(valdate, expiry)
n_strikes = 16
strike_percentiles = np.linspace(0.01, 0.99, n_strikes)
strike_conf = norm.ppf(strike_percentiles)

# Retrieve forward curve
fwd_curve = mdp.get_eq_forward_curves([name], valdate)[0]

# Retrieve local volatility
cvol = 0.40
cf_vols = [cvol] * n_strikes
lv = ConstantLocalVol(cvol)
lv = lvf.get_local_vols([name], valdate)[0]

# Specify strikes using LV variance at ATM
lv_vol = lv.path_vol(expiry_time, 0.0)
print(f"LV vol: {lv_vol}")
lv_stdev = lv_vol * np.sqrt(expiry_time)
fwd = fwd_curve.value(expiry)
strikes = fwd * np.exp(-0.5 * lv_stdev**2 + strike_conf * lv_stdev)

# Specify 2 different PDE configs
pde_config1 = PdeConfig(n_timesteps=500, n_meshes=500, n_stdevs=7, scheme='Rannacher')
pde_config2 = PdeConfig(n_timesteps=500, n_meshes=5000, n_stdevs=8, scheme='Rannacher')

# # Set up 2 different NumericalImpliedVol
# num_iv1 = NumericalImpliedVol(lv, pde_config=pde_config1)
# num_iv2 = NumericalImpliedVol(lv, pde_config=pde_config2)

# Calculate vanilla prices for both (straddles)
price1 = fpde.price_vanillas(valdate, expiry, strikes, fwd_curve, lv, pde_config=pde_config1)
price2 = fpde.price_vanillas(valdate, expiry, strikes, fwd_curve, lv, pde_config=pde_config2)

# # Calculate implied vols and RMSE
# call1 = (price1 + fwd - strikes) / 2.0
# call2 = (price2 + fwd - strikes) / 2.0
# ivol1 = black.implied_vols(expiry_time, strikes, True, fwd, call1)
# ivol2 = black.implied_vols(expiry_time, strikes, True, fwd, call2)
# rmse = metrics.rmse(ivol1, ivol2) * 10000.0
# rmse1 = metrics.rmse(ivol1, cf_vols) * 10000.0
# rmse2 = metrics.rmse(ivol2, cf_vols) * 10000.0

# Monte-Carlo simulation
n_steps, n_paths = 500, 100000
# mc_prices = mc.price_vanilla_surface(valdate, [expiry], [strikes], name, lv=lv, n_paths=n_paths,
#                                      n_timesteps=n_steps, constr_type='brownianbridge', rng_type='sobol')
# mc_prices = mc_prices[0]
# print(np.asarray(mc_prices))
# mc_prices = [64.970893, 50.252192, 43.791968, 39.453254, 36.317645, 34.06612, 32.573497, 31.810085,
#              31.814460, 32.695429, 34.657992, 38.071986, 43.645423, 52.94587, 70.678862, 137.537688]
mc_prices = [45.7008027,  33.89507105, 29.38515816, 26.55443066, 24.59816048, 23.23010422,
 22.33134773, 21.86353393, 21.8346015,  22.33448606, 23.4916185,  25.46680935,
 28.55667517, 33.44972718, 42.23371524, 72.28982854]

# Closed-form
cf_prices = black.price_straddles(np.full_like(strikes, expiry_time), strikes, fwd, cf_vols)

# Compare 6 smiles
# print(price1 / fwd)
# print(price2 / fwd)
# print(mc_prices / fwd)
# print(ivol1)
# print(ivol2)
# # print(f"Diff: {10000.0 * (ivol2 - ivol1)}")
# print(f"RMSE: {rmse}")
print(f"RMSE(PDE1-CF): {metrics.rmse(price1/fwd, cf_prices/fwd) * 10000.0}")
print(f"RMSE(PDE2-CF) {metrics.rmse(price2/fwd, cf_prices/fwd) * 10000.0}")
print(f"RMSE(PDE2-PDE1): {metrics.rmse(price2/fwd, price1/fwd) * 10000.0}")
print(f"RMSE(MC-CF): {metrics.rmse(mc_prices/fwd, cf_prices/fwd) * 10000.0}")
print(f"RMSE(PDE2-MC): {metrics.rmse(price2/fwd, mc_prices/fwd) * 10000.0}")
