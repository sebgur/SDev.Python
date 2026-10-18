import numpy as np
import matplotlib.pyplot as plt
from sdevpy.volatility.impliedvol.models import sabr
from sdevpy.analytics import bachelier as bc
from sdevpy.analytics import black as bs
from sdevpy.maths import metrics


############################## TODO ###############################################################
# * Shift the forward by some bps
# * Recalculate the alpha for Sticky-SABR
# * Recalculate the shifted smiles for sticky sabr and moneyness
# * Compare the deltas

# Define a shifted SABR smile
expiry = 2.5
fwd = 0.04
shift = 0.03
params = {'LnVol': 0.30, 'Beta': 0.2, 'Nu': 0.70, 'Rho': -0.30}
shifted_fwd = fwd + shift
min_spread, max_spread = -0.03, 0.08
min_k, max_k = fwd + min_spread, fwd + max_spread
n_points = 100
strike = np.linspace(min_k, max_k, n_points)
is_call = np.full(strike.shape, True)

# Calculate base smile
shifted_strike = strike + shift
base_slnvol = sabr.sabr_from_dict(expiry, shifted_strike, shifted_fwd, params)

# Calculate base normal vols
base_price = bs.price(expiry, shifted_strike, is_call, shifted_fwd, base_slnvol)
base_nvol = bc.implied_vol(expiry, strike, is_call, fwd, base_price)

# Check prices
# check_price = bc.price(expiry, strike, is_call, fwd, base_nvol)
# check_diff = metrics.bps_rmse(check_price, base_price) * 10000.0
# print(f"Check diff: {check_diff}")
# print(base_price)
# print(check_price)

# Calculate ATM normal vol to check the bump later
atm_slnvol = sabr.sabr_from_dict(expiry, shifted_fwd, shifted_fwd, params)
atm_price = bs.price(expiry, shifted_fwd, True, shifted_fwd, atm_slnvol)
atm_nvol = bc.implied_vol(expiry, fwd, True, fwd, atm_price)

# Bump the forward and recalculate alpha
bump = 50 / 10000.0
bumped_fwd = fwd + bump



# Plot
print(f"Normal ATM: {atm_nvol}")
# print(base_slnvol.shape)
# print(strike.shape)
# print(is_call.shape)

fix, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].plot(strike, base_slnvol)
axs[1].plot(strike, base_nvol)
plt.tight_layout()
plt.show()
