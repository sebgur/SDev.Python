""" Calculate PV and Greeks for an arbitrary number of dimensions and compare performance
    of bumps vs AAD as the dimension increases. """
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt

# Single simulation for PV
# Bumps
# Closed-form for PV, delta and gamma
# AAD MC
# Comparison
# Can we do a loop and draw the evolution chart in dimension?
# Give the choice to do second order or not
# Write a loop that outputs all its results to text at every iteration to avoid
#   losing the data


############### Runtime configuration #############################################################
DIM = 2
NUM_MC = 10
DTYPE = 'float64'
SEED1 = 1234
SEED2 = 42

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

SPOT = rng_init.uniform(SPOT_MIN, SPOT_MAX, DIM).reshape(-1, 1)
VOL = rng_init.uniform(VOL_MIN, VOL_MAX, DIM).reshape(-1, 1)
DIV = rng_init.uniform(DIV_MIN, DIV_MAX, DIM).reshape(-1, 1)

g_mean = np.zeros(shape=(DIM))
# Correlation matrix
correlation = 0.50
correl = np.ones((DIM, DIM), dtype=DTYPE) * correlation
for i in range(DIM):
    correl[i, i] = 1.0
# Covariance matrix
cov = correl.copy()
for i in range(len(correl)):
    for j in range(len(correl)):
        cov[i, j] *= VOL[i] * VOL[j]

# Payoff
STRIKE = 1.0
MATURITY = 2.5
TYPE = "Put"
W = -1.0 if TYPE == "Put" else 1.0

# Calculate deterministic quantities
STDEV = VOL * np.sqrt(MATURITY)
VAR = STDEV**2
DF = np.exp(-RATE * MATURITY)

print(SPOT)
print(VOL)
print(DIV)
print(correl)
print(cov)
print(W)
print(STDEV)

############### Bumps #############################################################################
def pv_mc(spot, gaussians):
    # Calculate deterministic forward
    fwd = spot * np.exp((RATE - DIV) * MATURITY)
    print(fwd)

    # Calculate final spot paths
    # future_spot = fwd * np.exp(-0.5 * VAR  + STDEV * gaussians)
    # print(future_spot)

    # # Calculate discounted payoff
    # perf = future_spot / fixings - 1.0
    # wperf = tf.reduce_min(perf, axis=1, keepdims=True)
    # floored_rate = floor + tf_smooth_max(wperf, floor)
    # eff_rate = cap - tf_smooth_max(-floored_rate, -cap)
    # payoff = df * notional * eff_rate

    # # Reduce
    # pv = tf.reduce_mean(payoff, axis=0)

    # return pv.numpy()[0]


g = rng_mc.multivariate_normal(g_mean, correl, size=NUM_MC)
print(g)
pv_mc(SPOT, g)


############### AAD ###############################################################################


############### CF ###############################################################


############### Numerical results #################################################################
