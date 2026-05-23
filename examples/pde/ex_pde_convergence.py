import datetime as dt
import numpy as np
from scipy.stats import norm
from sdevpy.market.eqforward import get_forward_curves
from sdevpy.volatility.localvol import localvol_factory as lvf
from sdevpy.volatility.localvol.localvol import ConstantLocalVol
from sdevpy.volatility.impliedvol.numerical_impliedvol import NumericalImpliedVol, DFLT_PDE_CONFIG
from sdevpy.pde.pdeschemes import PdeConfig
from sdevpy.pde import forwardpde as fpde
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.maths import metrics
from sdevpy.analytics import black


#################### TODO ###########################################
# * Turn on lognormal_density from LV
# * Add comparison to MC simulation
# * See if we can get accuracy that's similar to ConstantLocalVol
# * If moving to lognormal_density from LV, need to include the
#   initialization step in the optimizer.
# * Move to vectorize straddle payoff calibration (see code review)

# Specify underlying and date
name, valdate = "ABC", dt.datetime(2025, 12, 15)

# Specify products
tenor = '1y'
expiry = dts.advance(valdate, tenor)
expiry_time = timegrids.model_time(valdate, expiry)
strike_percentiles = np.linspace(0.01, 0.99, 16)
strike_conf = norm.ppf(strike_percentiles)

# Retrieve forward curve
fwd_curve = get_forward_curves([name], valdate)[0]
fwd = fwd_curve.value(expiry)

# Retrieve local volatility
lv = lvf.get_local_vols([name], valdate)[0]
cvol = 0.40
lv = ConstantLocalVol(cvol)

# Estimate strikes using LV variance at ATM
lv_vol = lv.path_vol(expiry_time, 0.0)
print(f"LV vol: {lv_vol}")
lv_stdev = lv_vol * np.sqrt(expiry_time)
strikes = fwd * np.exp(-0.5 * lv_stdev**2 + strike_conf * lv_stdev)
# print(f"Forward: {fwd}")
# print(f"Strikes: {strikes}")

# Specify 2 different PDE configs
pde_config1 = PdeConfig(n_timesteps=200, n_meshes=1000, mesh_vol=0.30, scheme='Rannacher',
                        percentile=1e-8)
pde_config2 = PdeConfig(n_timesteps=200, n_meshes=500, mesh_vol=0.50, scheme='Rannacher',
                        percentile=1e-6)

# # Set up 2 different NumericalImpliedVol
# num_iv1 = NumericalImpliedVol(lv, pde_config=pde_config1)
# num_iv2 = NumericalImpliedVol(lv, pde_config=pde_config2)

# Calculate vanilla prices for both (straddles)
price1 = fpde.price_vanillas(valdate, expiry, strikes, fwd_curve, lv, pde_config=pde_config1)
price2 = fpde.price_vanillas(valdate, expiry, strikes, fwd_curve, lv, pde_config=pde_config2)

# Calculate implied vols and RMSE
# (straddle + forward - strike) / 2 = call
call1 = (price1 + fwd - strikes) / 2.0
call2 = (price2 + fwd - strikes) / 2.0
ivol1 = black.implied_vols(expiry_time, strikes, True, fwd, call1)
ivol2 = black.implied_vols(expiry_time, strikes, True, fwd, call2)
rmse = metrics.rmse(ivol1, ivol2) * 10000.0
rmse1 = metrics.rmse(ivol1, [cvol]*16) * 10000.0
rmse2 = metrics.rmse(ivol2, [cvol]*16) * 10000.0

# Monte-Carlo simulation

# Compare 6 smiles
print(ivol1)
print(ivol2)
print(f"Diff: {10000.0 * (ivol2 - ivol1)}")
print(f"RMSE: {rmse}")
print(f"RMSE1 {rmse1}")
print(f"RMSE2 {rmse2}")
