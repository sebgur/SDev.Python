import numpy as np
import matplotlib.pyplot as plt
from sdevpy.volatility.impliedvol.models import sabr
from sdevpy.analytics import bachelier as bc
from sdevpy.analytics import black as bs
from sdevpy.maths import metrics


ONE_BP = 1.0 / 10_000.0


# Define a shifted SABR smile
expiry = 1.5
fwd = 0.04
shift = 0.03
params = {'LnVol': 0.30, 'Beta': 0.2, 'Nu': 0.70, 'Rho': -0.30}
min_spread, max_spread = -200, 200
n_points = 100
bump = 20 * ONE_BP

# Define test quantities
min_k, max_k = fwd + min_spread * ONE_BP, fwd + max_spread * ONE_BP
shifted_fwd = fwd + shift
strike = np.linspace(min_k, max_k, n_points)
is_call = np.full(strike.shape, True)
beta, nu, rho = params['Beta'], params['Nu'], params['Rho']

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

# Bump the forward and
bumped_fwd = fwd + bump
bumped_shifted_fwd = bumped_fwd + shift

# Reoptimize alpha such that the new ATM normal vol (i.e. the ATM normal vol with the new forward)
# is the same as the previous ATM normal vol (i.e. atm_nvol), given the parameters beta, nu and
# rho from the original SABR
bumped_alpha = sabr.alpha_from_atm_nvol(expiry, bumped_shifted_fwd, atm_nvol, beta, nu, rho)
bumped_lnvol = bumped_alpha / bumped_shifted_fwd ** (1.0 - beta)
bumped_params = {'LnVol': bumped_lnvol, 'Beta': 0.2, 'Nu': 0.70, 'Rho': -0.30}

# Check new parameters
bumped_atm_slnvol = sabr.sabr_from_dict(expiry, bumped_shifted_fwd, bumped_shifted_fwd, bumped_params)
bumped_atm_price = bs.price(expiry, bumped_shifted_fwd, True, bumped_shifted_fwd, bumped_atm_slnvol)
bumped_atm_nvol = bc.implied_vol(expiry, bumped_fwd, True, bumped_fwd, bumped_atm_price)

# Calculate new smile in sticky-SABR
bumped_slnvol = sabr.sabr_from_dict(expiry, shifted_strike, bumped_shifted_fwd, bumped_params)
ssabr_price = bs.price(expiry, shifted_strike, is_call, bumped_shifted_fwd, bumped_slnvol)
ssabr_nvol = bc.implied_vol(expiry, strike, is_call, bumped_fwd, ssabr_price)

# Calculate new smile in sticky-moneyness: for the requested strikes, the normal vols are those
# that the old surface would say at strikes of the same moneyness
new_moneyness = strike - bumped_fwd
old_strike = fwd + new_moneyness
old_shifted_strike = old_strike + shift
old_slnvol = sabr.sabr_from_dict(expiry, old_shifted_strike, shifted_fwd, params)
old_price = bs.price(expiry, old_shifted_strike, is_call, shifted_fwd, old_slnvol)
sbc_nvol = bc.implied_vol(expiry, old_strike, is_call, fwd, old_price)
diff_nvol = metrics.bps_rmse(ssabr_nvol, sbc_nvol)
sbc_price = bc.price(expiry, strike, is_call, bumped_fwd, sbc_nvol)

# Calculate deltas (infinitesimal)
ssabr_delta = (ssabr_price - base_price) / bump
sbc_delta = (sbc_price - base_price) / bump

# View results
print(f"Normal ATM: {atm_nvol*100:.4f}")
print(f"Normal ATM(after bump): {bumped_atm_nvol*100:.4f}")
print(f"Smile diff: {diff_nvol}")
# for s, ssabr_d, sbc_d in zip(strike, ssabr_delta, sbc_delta, strict=True):
#     print(f"Strike/Stick-SABR/Stick-BC: {s}/{ssabr_d}/{sbc_d}")

# Plot
if True:
    fix, axs = plt.subplots(1, 2, figsize=(10, 8))
    axs[0].plot(strike*100, base_nvol, color='blue', label='Base')
    axs[0].plot(strike*100, ssabr_nvol, color='red', label='Sticky SABR')
    axs[0].plot(strike*100, sbc_nvol, color='green', label='Sticky Bachelier')
    axs[0].legend(loc='best')
    axs[0].set_title('Normal vol')
    axs[1].plot(strike*100, ssabr_delta, color='red', label='Sticky SABR')
    axs[1].plot(strike*100, sbc_delta, color='green', label='Sticky Bachelier')
    axs[1].legend(loc='best')
    axs[1].set_title('Delta')
    plt.tight_layout()
    plt.show()
