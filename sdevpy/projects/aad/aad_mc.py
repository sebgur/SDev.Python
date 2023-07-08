""" Calculate PV and Greeks of an option basket using AAD on Monte-Carlo simulation """
import time as tm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sdevpy.montecarlo import smoothers
from sdevpy import settings
from sdevpy.analytics import black


# ################ Runtime configuration ##########################################################
# Parameters
EXPIRY = 2
SPOT = 100.0
VOL = 0.20
RATE = 0.04
DIV = 0.01
STRIKE = 100.0

# Bumps for differentials
SPOT_BUMP = 0.01  # Percentage of spot
VOL_BUMP = 0.01  # Percentage of vol
TIME_BUMP = 1.0 / 365.0
RATE_BUMP = 1.0 / 10000.0  # In bps

# Scalings
VEGA_SCL = settings.VEGA_SCALING
THETA_SCL = settings.THETA_SCALING
DV01_SCL = settings.DV01_SCALING
VOLGA_SCL = settings.VOLGA_SCALING
VANNA_SCL = settings.VANNA_SCALING

# Random generator seed
SEED = 42

# Use smoother in standard MC (always used in AAD)
USE_SMOOTHER_STD = False

# ################ Helper functions ###############################################################
def print_val(values_, is_pv_vector=False):
    """ Simple screen printing of valuation results """
    if is_pv_vector:
        print(f"PV: {values_[0][0]:,.4f}")
    else:
        print(f"PV: {values_[0]:,.4f}")
    print(f"Delta: {values_[1]:,.4f}")
    print(f"Gamma: {values_[2]:,.4f}")
    print(f"Vega: {values_[3]:,.4f}")
    print(f"Theta: {values_[4]:,.4f}")
    print(f"DV01: {values_[5]:,.4f}")
    print(f"Volga: {values_[6]:,.4f}")
    print(f"Vanna: {values_[7]:,.4f}")

# ################ Standard Monte-Carlo #########################################################
# Standard simulator
def simulate_std(spot_, vol_, time_, rate_, gaussians):
    """ Standard MC simulation for PV """
    # Calculate deterministic forward
    fwd = spot_ * np.exp((rate_ - DIV) * time_)

    # Calculate final spot paths
    stdev = vol_ * np.sqrt(time_)
    future_spot = fwd * np.exp(-0.5 * stdev * stdev + stdev * gaussians)

    # Calculate discounted payoff
    df = np.exp(-rate_ * time_)
    if USE_SMOOTHER_STD:
        payoff = df * smoothers.smooth_call(future_spot, STRIKE)  # Use smoothing
    else:
        payoff = df * np.maximum(future_spot - STRIKE, 0)

    # Reduce
    pv = np.mean(payoff, axis=0)

    return pv

# Sensitivities by bumps
def calculate_std(spot_, vol_, time_, rate_, num_mc):
    """ Calculate PV and sensitivities by bumps on MC """
    rng = np.random.RandomState(SEED)
    # Calculate gaussians only once
    gaussians = rng.normal(0.0, 1.0, (num_mc, 1))

    pv = simulate_std(spot_, vol_, EXPIRY, rate_, gaussians)

    # Delta-Gamma
    spot_u = spot_ * (1.0 + SPOT_BUMP)
    pv_spot_up = simulate_std(spot_u, vol_, time_, rate_, gaussians)

    spot_d = spot_ * (1.0 - SPOT_BUMP)
    pv_spot_down = simulate_std(spot_d, vol_, time_, rate_, gaussians)

    # Vega-Volga
    vol_u = vol_ * (1.0 + VOL_BUMP)
    pv_vol_up = simulate_std(spot_, vol_u, time_, rate_, gaussians)

    vol_d = vol_ * (1.0 - VOL_BUMP)
    pv_vol_down = simulate_std(spot_, vol_d, time_, rate_, gaussians)

    # Theta
    pv_expiry_down = simulate_std(spot_, vol_, time_ - TIME_BUMP, rate_, gaussians)

    # DV01
    pv_rate_up = simulate_std(spot_, vol_, time_, rate_ + RATE_BUMP, gaussians)
    pv_rate_down = simulate_std(spot_, vol_, time_, rate_ - RATE_BUMP, gaussians)

    # Vanna
    pv_uu = simulate_std(spot_u, vol_u, time_, rate_, gaussians)
    pv_ud = simulate_std(spot_u, vol_d, time_, rate_, gaussians)
    pv_du = simulate_std(spot_d, vol_u, time_, rate_, gaussians)
    pv_dd = simulate_std(spot_d, vol_d, time_, rate_, gaussians)

    # FDM
    delta = (pv_spot_up - pv_spot_down) / (2.0 * SPOT_BUMP * spot_)
    gamma = (pv_spot_up + pv_spot_down - 2.0 * pv) / np.power(SPOT_BUMP * spot_, 2)
    vega = (pv_vol_up - pv_vol_down) / (2.0 * VOL_BUMP * vol_) * VEGA_SCL
    theta = (pv_expiry_down - pv) / TIME_BUMP * THETA_SCL
    dv01 = (pv_rate_up - pv_rate_down) / (2.0 * RATE_BUMP) * DV01_SCL
    volga_size = np.power(VOL_BUMP * vol_, 2)
    volga = (pv_vol_up + pv_vol_down - 2.0 * pv) / volga_size * VOLGA_SCL
    vanna_size = 4.0 * SPOT_BUMP * spot_ * VOL_BUMP * vol_
    vanna = (pv_uu - pv_ud - pv_du + pv_dd) / vanna_size * VANNA_SCL

    return [pv, delta, gamma, vega, theta, dv01, volga, vanna]

# Test bumps
NUM_MC = 100 * 1000
values = calculate_std(SPOT, VOL, EXPIRY, RATE, NUM_MC)

print_val([v[0] for v in values])

# ################ AAD Monte-Carlo ###############################################################
# AAD 2nd order simulator with smoothing
def calculate_aad(spot_, vol_, time_, rate_, num_mc):
    """ Calculate PV and sensitivities by AAD MC """
    rng = np.random.RandomState(SEED)
    gaussians = rng.normal(0.0, 1.0, (num_mc, 1))

    tf_spot = tf.convert_to_tensor(spot_, dtype='float32')
    tf_vol = tf.convert_to_tensor(vol_)
    tf_time = tf.convert_to_tensor(time_, dtype='float32')
    tf_rate = tf.convert_to_tensor(rate_, dtype='float32')
    tf_div = tf.constant(DIV)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([tf_spot, tf_vol, tf_time, tf_rate])
        with tf.GradientTape(persistent=True) as tape2nd:
            tape2nd.watch([tf_spot, tf_vol])

            # Calculate deterministic forward
            fwd = tf_spot * tf.math.exp((tf_rate - tf_div) * tf_time)

            # Calculate final spot paths
            stdev = tf_vol * tf.math.sqrt(tf_time)
            future_spot = fwd * tf.math.exp(-0.5 * stdev * stdev + stdev * gaussians)

            # Calculate discounted payoff
            df = tf.math.exp(-tf_rate * tf_time)
            payoff = df * smoothers.tf_smooth_call(future_spot, STRIKE)

            # Reduce
            pv = tf.reduce_mean(payoff, axis=0)

        # Calculate delta and vega
        g_delta = tape2nd.gradient(pv, tf_spot)
        g_vega = tape2nd.gradient(pv, tf_vol)

    delta = tape.gradient(pv, tf_spot)
    gamma = tape.gradient(g_delta, tf_spot)
    vega = tape.gradient(pv, tf_vol)
    theta = tape.gradient(pv, tf_time)
    dv01 = tape.gradient(pv, tf_rate)
    volga = tape.gradient(g_vega, tf_vol)
    vanna = tape.gradient(g_delta, tf_vol)

    # Scale
    vega = vega.numpy() * VEGA_SCL
    theta = -theta.numpy() * THETA_SCL
    dv01 = dv01.numpy() * DV01_SCL
    volga = volga.numpy() * VOLGA_SCL
    vanna = vanna.numpy() * VANNA_SCL

    return [pv.numpy(), delta.numpy(), gamma.numpy(), vega, theta, dv01, volga, vanna]

# Test AAD with smoothing
NUM_MC = 100 * 1000
values = calculate_aad(SPOT, VOL, EXPIRY, RATE, NUM_MC)

print_val(values, is_pv_vector=True)

############# Test against CF #####################################################################
# Calculate with closed-form
values = black.price_and_greeks(EXPIRY, STRIKE, SPOT, VOL, RATE, DIV)
print_val(values)

######### Numerical Tests #########################################################################
# Ladder tests
NUM_POINTS = 100  # Number of loops/points in the charts
MIN_S = 20.0
MAX_S = 200.0
spots = np.linspace(MIN_S, MAX_S, NUM_POINTS, dtype='double')  # spot ladder
print(f"Displaying {NUM_POINTS:,} points over spot range [{MIN_S:,}, {MAX_S:,}]")

# Vectorize for broadcasting
vols = np.ones(NUM_POINTS, dtype='float32') * VOL
times = np.ones(NUM_POINTS, dtype='float32') * EXPIRY
rates = np.ones(NUM_POINTS, dtype='float32') * RATE

# Bump method
print("Calculating with bumps...")
NUM_MC = 20 * 1000  # Number of simulations
print(f"Number of simulations: {NUM_MC:,}")
time_bmp = tm.time()
results_bmp = calculate_std(spots, vols, times, rates, NUM_MC)
time_bmp = tm.time() - time_bmp

# AAD
print("Calculating with AAD...")
NUM_MC = 20 * 1000  # Number of simulations
print(f"Number of simulations: {NUM_MC:,}")
time_aad = tm.time()
results_aad = calculate_aad(spots, vols, times, rates, NUM_MC)
time_aad = tm.time() - time_aad

# AD closed-form
print("Calculating with AD closed-form...")
time_adcf = tm.time()
results_adcf = black.price_and_greeks(times, STRIKE, spots, vols, rates, DIV)
time_adcf = tm.time() - time_adcf

print("Calculation complete!")
print(f'Runtime(Bumps): {time_bmp:.1f}s')
print(f'Runtime(AAD): {time_aad:.1f}s')
print(f'Runtime(AD-CF): {time_adcf:.1f}s')

#### Charts
results_bmp = np.array(results_bmp)
results_aad = np.array(results_aad)
results_adcf = np.array(results_adcf)

# Select viewing range
VIEW_MIN = MIN_S  # Choose larger to zoom in
VIEW_MAX = MAX_S  # Choose smaller to zoom in

view_range = [i for i in range(NUM_POINTS) if spots[i] >= VIEW_MIN and spots[i] <= VIEW_MAX]
start = view_range[0]
end = view_range[-1]

def plot_value(plt_idx, name, result_idx, legend_location):
    """ Plot 3-way comparison between standard MC, AAD MC and closed-form """
    plt.subplot(4, 2, plt_idx)
    plt.title(name)
    plt.xlabel('Spot')
    plt.plot(spots[start:end], results_bmp[result_idx][start:end], 'green', alpha=0.7,
             label='Bumps')
    plt.plot(spots[start:end], results_adcf[result_idx][start:end], color='blue', alpha=0.8,
             label='AD-CF')
    plt.plot(spots[start:end], results_aad[result_idx][start:end], color='red',
             label='AAD')
    plt.legend(loc=legend_location)

# Plot results
plt.ioff()
plt.figure(figsize=(15, 16))
plt.subplots_adjust(hspace=0.40, wspace=0.20)

plot_value(1, "PV", 0, 'upper left')
plot_value(2, "Delta", 1, 'upper left')
plot_value(3, "Vega", 3, 'upper right')
plot_value(4, "Gamma", 2, 'upper right')
plot_value(5, "Volga", 6, 'upper left')
plot_value(6, "Vanna", 7, 'upper right')
plot_value(7, "Theta", 4, 'upper right')
plot_value(8, "DV01", 5, 'lower right')

plt.show()
