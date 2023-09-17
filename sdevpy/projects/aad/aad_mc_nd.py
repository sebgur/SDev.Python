""" Calculate PV and Greeks for an arbitrary number of dimensions and compare performance
    of bumps vs AAD as the dimension increases. """
import numpy as np
import tensorflow as tf
from scipy.stats import norm
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.montecarlo import smoothers

# AAD MC
# Loop that outputs  its results to text at every iteration to avoid losing the data

############### Runtime configuration #############################################################
DIM = 2
NUM_MC = 1000#500 * 1000 #50000
DTYPE = 'float64'
SEED_INIT = 1234
SEED_MC = 42
SEED_AD = 4321#SEED_MC

BUMP = 0.01

tf.keras.backend.set_floatx(DTYPE)
rng_init = np.random.RandomState(SEED_INIT)
rng_mc = np.random.RandomState(SEED_MC)
rng_ad = np.random.RandomState(SEED_AD)

# Market data
RATE = 0.02
SPOT_MIN = 0.9
SPOT_MAX = 1.1
VOL_MIN = 0.15
VOL_MAX = 0.25
DIV_MIN = -0.01
DIV_MAX = 0.04

SPOT = rng_init.uniform(SPOT_MIN, SPOT_MAX, DIM)
VOL = rng_init.uniform(VOL_MIN, VOL_MAX, DIM)
DIV = rng_init.uniform(DIV_MIN, DIV_MAX, DIM)

# Random numbers
G_MEAN = np.zeros(shape=(DIM))
# Correlation matrix
CORR = 0.5
CORR_MATRIX = np.ones((DIM, DIM), dtype=DTYPE) * CORR
for i in range(DIM):
    CORR_MATRIX[i, i] = 1.0
# Covariance matrix
COV = CORR_MATRIX.copy()
for i in range(len(CORR_MATRIX)):
    for j in range(len(CORR_MATRIX)):
        COV[i, j] *= VOL[i] * VOL[j]

# Payoff
NUM_STRIKES = 50
STRIKE = np.linspace(0.1, 4.5, NUM_STRIKES)
# STRIKE = np.asarray([0.9, 1.0, 1.1])
MATURITY = 1.5
# TYPE = "Put"
TYPE = "Call"
W = -1.0 if TYPE == "Put" else 1.0
IS_CALL = False if TYPE == 'Put' else True

# Calculate deterministic quantities
STDEV = VOL * np.sqrt(MATURITY)
VAR = STDEV**2
DF = np.exp(-RATE * MATURITY)

print("SPOT\n", SPOT)

############### MC ################################################################################
def pv_mc(spot, strikes, gaussians):
    # Calculate deterministic forward
    fwd = spot * np.exp((RATE - DIV) * MATURITY)

    # Calculate final spot paths
    future_spot = fwd * np.exp(-0.5 * VAR  + STDEV * gaussians)

    # Calculate discounted payoff
    prod = np.prod(future_spot, axis=1, keepdims=True)
    if IS_CALL:
        diff = smoothers.smooth_max_diff(prod, strikes)
    else:
        diff = smoothers.smooth_max_diff(strikes, prod)

    ddiff = DF * diff

    # Reduce
    pv = np.sum(ddiff, axis=0) / NUM_MC

    return pv

def bump_spot(spot, bump, index):
    b_spot = spot.copy()
    b_spot[index] *= (1.0 + bump)
    return b_spot

def greeks_mc(spot, strikes, gaussians, pv, calc_gammas):
    """ Calculate all bumps """
    num_strikes = strikes.shape[0]
    delta = np.ndarray(shape=(DIM, num_strikes))
    gamma = None if calc_gammas is False else np.ndarray(shape=(DIM, DIM, num_strikes)) 

    # Single bumps
    for idx in range(DIM):
        # Bumps up
        pv_u = pv_mc(bump_spot(spot, BUMP, idx), strikes, gaussians)
        shift_i = spot[idx] * BUMP

        if calc_gammas is False:
            delta[idx] = (pv_u - pv) / shift_i
        else:
            # Bumps down for gammas (one-sided deltas otherwise)
            pv_d = pv_mc(bump_spot(spot, -BUMP, idx), strikes, gaussians)
            delta[idx] = (pv_u - pv_d) / (2.0 * shift_i)
            gamma[idx, idx] = (pv_u + pv_d - 2.0 * pv) / np.power(shift_i, 2.0)

            # Crosses
            for j in range(idx + 1, DIM):
                shift_j = spot[j] * BUMP
                spot_uu = bump_spot(bump_spot(spot, BUMP, idx), BUMP, j)
                spot_ud = bump_spot(bump_spot(spot, BUMP, idx), -BUMP, j)
                spot_du = bump_spot(bump_spot(spot, -BUMP, idx), BUMP, j)
                spot_dd = bump_spot(bump_spot(spot, -BUMP, idx), -BUMP, j)
                pv_uu = pv_mc(spot_uu, strikes, gaussians)
                pv_ud = pv_mc(spot_ud, strikes, gaussians)
                pv_du = pv_mc(spot_du, strikes, gaussians)
                pv_dd = pv_mc(spot_dd, strikes, gaussians)
                gamma[idx, j] = (pv_uu - pv_ud - pv_du + pv_dd) / (4.0 * shift_i * shift_j)

    if calc_gammas: # symmetric crosses
        for i in range(DIM):
            for j in range(0, i):
                gamma[i, j] = gamma[j, i]

    return delta, gamma

def value_mc(spot, strikes, gaussians, calc_gammas):
    pv = pv_mc(spot, strikes, gaussians)
    delta, gamma = greeks_mc(spot, strikes, gaussians, pv, calc_gammas)
    return pv, delta, gamma

# Trigger MC valuation
g = rng_mc.multivariate_normal(G_MEAN, CORR_MATRIX, size=NUM_MC)
mc_pv, mc_delta, mc_gamma = value_mc(SPOT, STRIKE, g, calc_gammas=True)
print("MC-PV\n", mc_pv)
print("MC-Delta\n", mc_delta)
# print("MC-Gamma\n", mc_gamma)

############### CF ################################################################################
def value_cf(spot, strikes):
    fwd = spot * np.exp((RATE - DIV) * MATURITY)
    prod_fwd = np.prod(fwd)
    svar = np.sum(VAR)
    scov = np.sum(COV)
    fwd_correc = (scov * MATURITY - svar) / 2.0
    prod_fwd = prod_fwd * np.exp(fwd_correc)
    prod_vol = np.sqrt(scov)

    # PV
    pv = DF * black.price(MATURITY, strikes, IS_CALL, prod_fwd, prod_vol)

    # Delta
    delta = np.ndarray(shape=(DIM, strikes.shape[0]))
    prod_stdev = prod_vol * np.sqrt(MATURITY)
    d1 = np.log(prod_fwd / strikes) / prod_stdev + 0.5 * prod_stdev
    nd1 = norm.cdf(W * d1)
    for index in range(DIM):
        delta[index] = DF * W * (prod_fwd / spot[index]) * nd1 

    # Gamma
    gamma = np.ndarray(shape=(DIM, DIM, strikes.shape[0]))
    ndf1s = norm.pdf(d1) / prod_stdev

    # Original
    for idx1 in range(DIM):
        for idx2 in range(idx1, DIM):
            coeff = DF * prod_fwd / spot[idx1] / spot[idx2]
            gamma_ = ndf1s.copy()
            if idx1 != idx2:
                gamma_ += W * nd1
            gamma_ *= coeff
            gamma[idx1, idx2] = gamma_

    for idx1 in range(DIM):
        for idx2 in range(0, idx1):
            gamma[idx1, idx2] = gamma[idx2, idx1]


    return pv, delta, gamma

# Trigger CF valuation
cf_pv, cf_delta, cf_gamma = value_cf(SPOT, STRIKE)
# print("CF-PV\n", cf_pv)
# print("CF-Delta\n", cf_delta)
# print("CF-Gamma\n", cf_gamma)

############### AD ################################################################################
def value_ad(spot, strikes, gaussians, calc_gammas):
    # Convert to tensors
    tf_spot = tf.convert_to_tensor(spot, dtype=DTYPE)
    tf_var = tf.constant(VAR, dtype=DTYPE)
    tf_stdev = tf.constant(STDEV, dtype=DTYPE)
    tf_time = tf.constant(MATURITY, dtype=DTYPE)
    tf_rate = tf.constant(RATE, dtype=DTYPE)
    tf_div = tf.constant(DIV, dtype=DTYPE)
    tf_strikes = tf.constant(strikes, dtype=DTYPE)
  
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([tf_spot])

        # Calculate deterministic forward
        fwd = tf_spot * tf.math.exp((tf_rate - tf_div) * tf_time)

        # Calculate final spot paths
        future_spot = fwd * tf.math.exp(-0.5 * tf_var  + tf_stdev * gaussians)

        # Calculate discounted payoff
        prod = tf.math.reduce_prod(future_spot, axis=1, keepdims=True)
        if IS_CALL:
            diff = smoothers.tf_smooth_max_diff(prod, tf_strikes)
        else:
            diff = smoothers.tf_smooth_max_diff(tf_strikes, prod)

        ddiff = DF * diff

        # Reduce
        pv = tf.math.reduce_sum(ddiff, axis=0) / NUM_MC


    # with tf.GradientTape(persistent=True) as tape:
    #     tape.watch([tf_spot])
    #     with tf.GradientTape(persistent=True) as tape2nd:
    #         tape2nd.watch([tf_spot])

    #         # Calculate deterministic forward
    #         fwd = tf_spot * tf.math.exp((tf_rate - tf_div) * tf_time)

    #         # Calculate final spot paths
    #         stdev = tf_vol * tf.math.sqrt(tf_time)
    #         future_spot = fwd * tf.math.exp(-0.5 * stdev * stdev + stdev * gaussians)

    #         # Calculate discounted payoff
    #         df = tf.math.exp(-tf_rate * tf_time)
    #         payoff = df * tf_smooth_max(future_spot, strikes)

    #         # Reduce
    #         pv = tf.reduce_mean(payoff, axis=0)

    #     # Calculate delta and vega
    #     g_delta = tape2nd.gradient(pv, tf_spot)
       
    # delta = tape.gradient(pv, tf_spot)
    # gamma = tape.gradient(g_delta, tf_spot)

    delta = tape.jacobian(pv, tf_spot)
    gamma = None
    delta = delta.numpy().transpose()
    # print(delta.shape)
    # tdelta = delta.transpose()
    # print(tdelta)

    return pv.numpy(), delta, gamma

# Trigger AD valuation
g = rng_ad.multivariate_normal(G_MEAN, CORR_MATRIX, size=NUM_MC)
ad_pv, ad_delta, ad_gamma = value_ad(SPOT, STRIKE, g, calc_gammas=True)
print("AD-PV\n", ad_pv)
print("AD-Delta\n", ad_delta)
print("AD-Gamma\n", ad_gamma)


############### Numerical results #################################################################
DIM1 = 0
DIM2 = 1
fig, axs = plt.subplots(3, 2, layout="constrained")
fig.suptitle("AD vs MC Bumps vs CF", size='x-large', weight='bold')
fig.set_size_inches(12, 8)

# PV
axs[0, 0].plot(STRIKE, mc_pv, color='red', label='MC')
axs[0, 0].plot(STRIKE, cf_pv, color='blue', label='CF')
axs[0, 0].plot(STRIKE, ad_pv, color='green', label='AD')
axs[0, 0].set_xlabel('Strike')
axs[0, 0].set_title("PV")
axs[0, 0].legend(loc='upper right')

# Delta
axs[1, 0].plot(STRIKE, mc_delta[DIM1], color='red', label='MC')
axs[1, 0].plot(STRIKE, cf_delta[DIM1], color='blue', label='CF')
axs[1, 0].plot(STRIKE, ad_delta[DIM1], color='green', label='AD')
axs[1, 0].set_xlabel('Strike')
axs[1, 0].set_title("Delta dimension " + str(DIM1))
axs[1, 0].legend(loc='upper right')

axs[1, 1].plot(STRIKE, mc_delta[DIM2], color='red', label='MC')
axs[1, 1].plot(STRIKE, cf_delta[DIM2], color='blue', label='CF')
axs[1, 1].plot(STRIKE, ad_delta[DIM2], color='green', label='AD')
axs[1, 1].set_xlabel('Strike')
axs[1, 1].set_title("Delta dimension " + str(DIM2))
axs[1, 1].legend(loc='upper right')

# Gamma
axs[2, 0].plot(STRIKE, mc_gamma[DIM1][DIM1], color='red', label='MC')
axs[2, 0].plot(STRIKE, cf_gamma[DIM1][DIM1], color='blue', label='CF')
axs[2, 0].set_xlabel('Strike')
axs[2, 0].set_title("Gamma dimension " + str(DIM1))
axs[2, 0].legend(loc='upper right')

axs[2, 1].plot(STRIKE, mc_gamma[DIM1][DIM2], color='red', label='MC')
axs[2, 1].plot(STRIKE, cf_gamma[DIM1][DIM2], color='blue', label='CF')
axs[2, 1].set_xlabel('Strike')
axs[2, 1].set_title("Cross-Gamma dimensions " + str(DIM1) + "/" + str(DIM2))
axs[2, 1].legend(loc='upper right')

plt.show()
