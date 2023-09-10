""" Calculate PV and Greeks for an arbitrary number of dimensions and compare performance
    of bumps vs AAD as the dimension increases. """
import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.montecarlo import smoothers

# Closed-form for delta, Compare with MC
# Closed-form for gamma, Compare with MC
# Pick 2 directions and display 7 matrices
# AAD MC
# Comparison
# Can we do a loop and draw the evolution chart in dimension?
# Give the choice to do second order or not
# Write a loop that outputs all its results to text at every iteration to avoid
#   losing the data

############### Runtime configuration #############################################################
DIM = 4
NUM_MC = 20000
DTYPE = 'float64'
SEED1 = 1234
SEED2 = 42

BUMP = 0.01

tf.keras.backend.set_floatx(DTYPE)
rng_init = np.random.RandomState(SEED1)
rng_mc = np.random.RandomState(SEED2)

# Market data
RATE = 0.02
SPOT_MIN = 0.9
SPOT_MAX = 1.1
VOL_MIN = 0.15
VOL_MAX = 0.40
DIV_MIN = -0.01
DIV_MAX = 0.04

SPOT = rng_init.uniform(SPOT_MIN, SPOT_MAX, DIM)#.reshape(1, -1)
VOL = rng_init.uniform(VOL_MIN, VOL_MAX, DIM)#.reshape(1, -1)
DIV = rng_init.uniform(DIV_MIN, DIV_MAX, DIM)#.reshape(1, -1)

g_mean = np.zeros(shape=(DIM))
# Correlation matrix
correlation = 0.5
correl = np.ones((DIM, DIM), dtype=DTYPE) * correlation
for i in range(DIM):
    correl[i, i] = 1.0
# Covariance matrix
COV = correl.copy()
for i in range(len(correl)):
    for j in range(len(correl)):
        COV[i, j] *= VOL[i] * VOL[j]

# Payoff
NUM_STRIKES = 50
STRIKE = np.linspace(0.1, 4.5, NUM_STRIKES)
# STRIKE = np.asarray([0.9, 1.0, 1.1])
MATURITY = 2.5
TYPE = "Put"
# TYPE = "Call"
W = -1.0 if TYPE == "Put" else 1.0
IS_CALL = False if TYPE == 'Put' else True

# Calculate deterministic quantities
STDEV = VOL * np.sqrt(MATURITY)
VAR = STDEV**2
DF = np.exp(-RATE * MATURITY)

print("SPOT\n", SPOT)
# print("VOL\n", VOL)
# print("DIV\n", DIV)
# print("correl\n", correl)
# print("cov\n", cov)
# print("W\n", W)
# print("VAR\n", VAR)
# print("STDEV\n", STDEV)

############### Bumps #############################################################################
def pv_mc(spot, gaussians, strikes):
    # Calculate deterministic forward
    fwd = spot * np.exp((RATE - DIV) * MATURITY)
    # print("fwd\n", fwd)

    # Calculate final spot paths
    # print(STDEV.shape)
    future_spot = fwd * np.exp(-0.5 * VAR  + STDEV * gaussians)
    # print("future_spot\n", future_spot)

    # Calculate discounted payoff
    prod = np.prod(future_spot, axis=1, keepdims=True)
    # print("prod\n", prod)
    diff = smoothers.smooth_max_diff(strikes, prod)
    # print("diff\n", diff)
    ddiff = DF * diff
    # print("discdiff\n", ddiff)

    # Reduce
    pv = np.sum(ddiff, axis=0) / NUM_MC

    return pv


# Trigger MC PV
g = rng_mc.multivariate_normal(g_mean, correl, size=NUM_MC)
mc_pv = pv_mc(SPOT, g, STRIKE)
# print("MC-PV\n", mc_pv)

def bump_spot(spot, bump, index):
    b_spot = spot.copy()
    b_spot[index] *= (1.0 + bump)
    return b_spot

def greeks_mc(gaussians, strikes, pv, calculate_gammas):
    """ Calculate all bumps """
    num_strikes = strikes.shape[0]
    delta = np.ndarray(shape=(DIM, num_strikes))
    gamma = None if calculate_gammas is False else np.ndarray(shape=(DIM, DIM, num_strikes)) 

    # Single bumps
    for i in range(DIM):
        # Bumps up
        pv_u = pv_mc(bump_spot(SPOT, BUMP, i), gaussians, strikes)
        shift_i = SPOT[i] * BUMP

        if calculate_gammas is False:
            delta[i] = (pv_u - pv) / shift_i
        else:
            # gamma = np.ndarray(shape=(DIM, DIM, num_strikes))
            # Bumps down for gammas (one-sided deltas otherwise)
            pv_d = pv_mc(bump_spot(SPOT, -BUMP, i), gaussians, strikes)
            delta[i] = (pv_u - pv_d) / (2.0 * shift_i)
            gamma[i, i] = (pv_u + pv_d - 2.0 * pv) / np.power(shift_i, 2.0)

            # Crosses
            for j in range(i + 1, DIM):
                shift_j = SPOT[j] * BUMP
                spot_uu = bump_spot(bump_spot(SPOT, BUMP, i), BUMP, j)
                spot_ud = bump_spot(bump_spot(SPOT, BUMP, i), -BUMP, j)
                spot_du = bump_spot(bump_spot(SPOT, -BUMP, i), BUMP, j)
                spot_dd = bump_spot(bump_spot(SPOT, -BUMP, i), -BUMP, j)
                pv_uu = pv_mc(spot_uu, gaussians, strikes)
                pv_ud = pv_mc(spot_ud, gaussians, strikes)
                pv_du = pv_mc(spot_du, gaussians, strikes)
                pv_dd = pv_mc(spot_dd, gaussians, strikes)
                gamma[i, j] = (pv_uu - pv_ud - pv_du + pv_dd) / (4.0 * shift_i * shift_j)

    if calculate_gammas: # symmetric crosses
        for i in range(DIM):
            for j in range(0, i):
                gamma[i, j] = gamma[j, i]

    return delta, gamma

# Trigger MC Greeks
mc_delta, mc_gamma = greeks_mc(g, STRIKE, mc_pv, calculate_gammas=True)
# print("MC-Delta\n", delta)
# print("MC-Gamma\n", gamma)

############### CF ################################################################################
N = norm.cdf
def pv_cf(spot, strikes):
    fwd = spot * np.exp((RATE - DIV) * MATURITY)
    prod_fwd = np.prod(fwd)
    svar = np.sum(VAR)
    scov = np.sum(COV)
    fwd_correc = (scov * MATURITY - svar) / 2.0
    prod_fwd = prod_fwd * np.exp(fwd_correc)
    prod_vol = np.sqrt(scov)

    pv = DF * black.price(MATURITY, strikes, IS_CALL, prod_fwd, prod_vol)

    delta = np.ndarray(shape=(DIM, strikes.shape[0]))
    prod_stdev = prod_vol * np.sqrt(MATURITY)
    d1 = np.log(prod_fwd / strikes) / prod_stdev + 0.5 * prod_stdev
    nd1 = N(W * d1)
    for index in range(DIM):
        delta[index] = DF * W * (prod_fwd / spot[index]) * nd1 

    return pv, delta

cf_pv, cf_delta = pv_cf(SPOT, STRIKE)
# print("CF-PV\n", cf_pv)

############### AAD ###############################################################################


############### Numerical results #################################################################
index1 = 2
index2 = 3
fig, axs = plt.subplots(2, 2, layout="constrained")
fig.suptitle("MC vs CF", size='x-large', weight='bold')
fig.set_size_inches(12, 8)
axs[0, 0].plot(STRIKE, mc_pv, color='red', label='MC')
axs[0, 0].plot(STRIKE, cf_pv, color='blue', label='CF')
axs[0, 0].set_xlabel('Strike')
axs[0, 0].set_title("PV")
axs[0, 0].legend(loc='upper right')

axs[1, 0].plot(STRIKE, mc_delta[index1], color='red', label='MC')
axs[1, 0].plot(STRIKE, cf_delta[index1], color='blue', label='CF')
axs[1, 0].set_xlabel('Strike')
axs[1, 0].set_title("Delta index 1")
axs[1, 0].legend(loc='upper right')

axs[1, 1].plot(STRIKE, mc_delta[index2], color='red', label='MC')
axs[1, 1].plot(STRIKE, cf_delta[index2], color='blue', label='CF')
axs[1, 1].set_xlabel('Strike')
axs[1, 1].set_title("Delta index 1")
axs[1, 1].legend(loc='upper right')


plt.show()
