# ########################## Imports and versions ##########################################################
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import time as tm
import scipy.stats
import sklearn as sk
from sklearn.metrics import mean_squared_error
import tensorflow_probability as tfp
from platform import python_version
import os
from datetime import datetime
import pandas as pd

print("Python version: " + python_version())
print("TensorFlow version: " + tf.__version__)
print("TF-Keras version: " + tf.keras.__version__)
print("TF-Probability version: " + tfp.__version__)
print("NumPy version: " + np.__version__)
print("SciPy version: " + scipy.__version__)
print("SciKit version: " + sk.__version__)

# ############################## Payoff and parameters #####################################################
# Payoff and model parameters
dtype = 'float64'
tf.keras.backend.set_floatx(dtype)
# 2D
# spot = np.asarray([100, 105], dtype=dtype)
# vol = np.asarray([0.25, 0.35], dtype=dtype)
# div = np.asarray([0.02, 0.01], dtype=dtype)
# fixings = np.asarray([90, 95], dtype=dtype)  # Past value of the spot to measure performance
# model_name = "model_2_assets"
# 3D
# spot = np.asarray([100, 105, 102], dtype=dtype)
# vol = np.asarray([0.25, 0.35, 0.30], dtype=dtype)
# div = np.asarray([0.02, 0.01, 0.01], dtype=dtype)
# fixings = np.asarray([90, 95, 92], dtype=dtype)  # Past value of the spot to measure performance
# model_name = "model_3_assets"
# 4D
# spot = np.asarray([100, 105, 102, 104], dtype=dtype)
# vol = np.asarray([0.25, 0.35, 0.30, 0.40], dtype=dtype)
# div = np.asarray([0.02, 0.01, 0.01, 0.03], dtype=dtype)
# fixings = np.asarray([90, 95, 92, 97], dtype=dtype)  # Past value of the spot to measure performance
# model_name = "model_4_assets"
# 5D
spot = np.asarray([100, 105, 102, 104, 103], dtype=dtype)
vol = np.asarray([0.25, 0.35, 0.30, 0.40, 0.30], dtype=dtype)
div = np.asarray([0.02, 0.01, 0.01, 0.03, 0.025], dtype=dtype)
fixings = np.asarray([90, 95, 92, 97, 102], dtype=dtype)  # Past value of the spot to measure performance
model_name = "model_5_assets"

rate = 0.015
expiry = 2.5
floor = -10.0 / 100  # -1.5 / 100  # Flooring the worst performance
cap = 10.0 / 100  # Capping the best performance
notional = 10000.0  # 10 * 1000
num_assets = len(spot)
correlation = 0.50
# Correlation matrix
correl = np.ones((num_assets, num_assets), dtype=dtype) * correlation
for i in range(num_assets):
    correl[i, i] = 1.0
# Covariance matrix
cov = correl.copy()
for i in range(len(correl)):
    for j in range(len(correl)):
        cov[i, j] *= vol[i] * vol[j]

# Random generator seed
seed = 42

# Model saving/loading
model_save_path = r"W:\\Models\\pinns_worst_of\\"
model_folder = os.path.join(model_save_path, model_name)

print("Number of assets: " + str(num_assets))


# ############################## Helper functions ##########################################################
# Print vector
def print_vec(name, data):
    print(name + ": [", end=""),
    for i_ in range(len(data)):
        print("{:,.2f}".format(data[i_]), end="")
        if i_ < len(data) - 1:
            print(", ", end="")
    print("]")


# Smoothing parameters (for the max function)
smooth_vol = 0.02
smooth_time = 10.0 / 365.0
smooth_stdev = smooth_vol * np.sqrt(smooth_time)
s2p = np.sqrt(2.0 * np.pi)
tf_N = tfp.distributions.Normal(0.0, 1.0)


# Tensorflow implementation of the payoff smoother ('tf_smooth_max')
def tf_approx_cdf(x):
    return 1.0 / (1.0 + tf.math.exp(-x / 0.5879))


# Smoother
def tf_smooth_max(x, y):
    d1 = (x - y) / smooth_stdev
    n1 = tf_approx_cdf(d1)  # Older
    return (x - y) * n1 + 0.5 * smooth_stdev * tf.math.exp(-0.5 * tf.math.pow(d1, 2)) / s2p


# ############################## Define boundaries #########################################################
# Time boundaries at 0 and expiry
tmin = tf.constant(0.0, dtype=dtype)
tmax = tf.constant(expiry, dtype=dtype)

# Spot boundaries defined as extreme percentiles of the distribution
N = scipy.stats.norm
fwdT = np.asarray(spot) * np.exp((rate - div) * expiry)
stdev = vol * np.sqrt(expiry)
conf = 0.9999  # Chosen percentile of the distributions (0.9999 worked quite well in 1D)
percentile = N.ppf(conf)
xmin = -0.5 * stdev * stdev - percentile * stdev
xmax = -0.5 * stdev * stdev + percentile * stdev
smin = fwdT * np.exp(xmin)
smax = fwdT * np.exp(xmax)

# Bounds in vector form (for input scaling)
lb = tf.concat([[tmin], xmin], axis=0)
ub = tf.concat([[tmax], xmax], axis=0)

print("Min time: {:,.2f}".format(tmin))
print("Max time: {:,.2f}".format(tmax))
print_vec("Spots", spot)
print_vec("Forwards", fwdT)
print_vec("Min spots", smin)
print_vec("Max spots", smax)
print_vec("Min red. spots", xmin)
print_vec("Max red. spots", xmax)


# ############################## Functions at boundaries ###################################################
# Payoff at maturity
def payoff(x):
    s = fwdT * tf.math.exp(x)
    perf = (s - fixings) / fixings
    # perf = s / fixings - 1.0  # Weird Python bug? This line gives inaccuracies, but the above is accurate
    wperf = tf.math.reduce_min(perf, axis=1, keepdims=True)
    floored_rate = tf.math.maximum(wperf, floor)
    eff_rate = tf.math.minimum(floored_rate, cap)
    return notional * eff_rate


# At extreme low values of the spots, the worst performance is below the floor
def lw_boundary(t, x):
    n = x.shape[0]
    df_ = tf.math.exp(-rate * (tmax - t))
    return notional * df_ * tf.ones((n, 1), dtype=dtype) * floor


# At extreme high values of the spots, the worst performance is above the cap
def up_boundary(t, x):
    n = x.shape[0]
    df_ = tf.math.exp(-rate * (tmax - t))
    return notional * df_ * tf.ones((n, 1), dtype=dtype) * cap


# ############################## Build the dataset #########################################################
def build_dataset(num_final_, nb_, num_pde_, init_rnd=False):
    if init_rnd:
        tf.random.set_seed(seed)

    # Draw payoff points
    t0_ = tf.ones((num_final_, 1), dtype=dtype) * tmax
    x0 = tf.random.uniform((num_final_, num_assets), xmin, xmax, dtype=dtype)
    payoff_points = tf.concat([t0_, x0], axis=1)
    payoff_values = payoff(x0)

    # Draw side boundaries points
    lw_tb = tf.random.uniform((nb_, 1), tmin, tmax, dtype=dtype)
    lw_xb = tf.ones((nb_, num_assets), dtype=dtype) * xmin
    up_tb = lw_tb  # We could also draw another set
    up_xb = tf.ones((nb_, num_assets), dtype=dtype) * xmax
    tb = tf.concat([lw_tb, up_tb], axis=0)
    xb = tf.concat([lw_xb, up_xb], axis=0)
    side_points = tf.concat([tb, xb], axis=1)
    lw_side_values = lw_boundary(lw_tb, lw_xb)
    up_side_values = up_boundary(up_tb, up_xb)
    side_values = tf.concat([lw_side_values, up_side_values], axis=0)

    # Pack boundary points together (payoff and side)
    if len(side_points) > 0:
        boundary_points_ = [payoff_points, side_points]
        boundary_values_ = [payoff_values, side_values]
    else:
        boundary_points_ = [payoff_points]
        boundary_values_ = [payoff_values]

    # Draw PDE points
    tpde = tf.random.uniform((num_pde_, 1), tmin, tmax, dtype=dtype)
    xpde = tf.random.uniform((num_pde_, num_assets), xmin, xmax, dtype=dtype)
    pde_points_ = tf.concat([tpde, xpde], axis=1)

    return pde_points_, boundary_points_, boundary_values_


# ############################## Model composition helpers #################################################
kinits = tf.keras.initializers
klayers = tf.keras.layers


def add_hidden_layer(model_, neurons_, activation_):
    init = kinits.glorot_normal

    model_.add(klayers.Dense(neurons_, activation=activation_, kernel_initializer=init,
                             use_bias=True, bias_initializer=kinits.Constant(0.1)))


# Set up the network architecture
def compose_model(hidden_layers_, num_neurons_):
    model_ = tf.keras.Sequential()

    # multi-d inputs
    model_.add(tf.keras.Input(1 + num_assets))

    # Scaling layer to map the inputs to (-1, 1)
    model_.add(klayers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0))

    # Hidden layers
    for hl in hidden_layers_:
        add_hidden_layer(model_, num_neurons_, hl)
        model_.add(klayers.Dropout(0.2))

    # 1d outputs
    model_.add(klayers.Dense(1))

    return model_


# ############################## PDE and loss ##############################################################
# Loss and gradient calculations
def calculate_pde(model_, pde_points_):
    with tf.GradientTape() as tape:
        # Split t and x to compute partial derivatives
        t, x = pde_points_[:, 0:1], pde_points_[:, 1:]
        tape.watch(x)  # We use this take for the 2nd order, so only need x
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([t, x])
            point = tf.concat([t, x], axis=1)
            # Calculate function
            u = model_(point)

        # Compute space gradient
        u_x = tape2.gradient(u, x)

        # Compute time differential
        u_t = tape2.gradient(u, t)

    # Use batch jacobian as there is no dependency across points
    u_xx = tape.batch_jacobian(u_x, x)

    del tape2
    del tape

    # Calculate residual of the Black-Scholes PDE
    order1 = -0.5 * tf.reduce_sum(tf.math.pow(vol, 2) * u_x, axis=1, keepdims=True)
    order2 = 0.5 * tf.reduce_sum(tf.reduce_sum(cov * u_xx, axis=1, keepdims=True), axis=2, keepdims=True)

    return u_t + order1 + order2[:, 0] - rate * u


def calculate_loss(model_, pde_points_, boundary_points_, boundary_values_):
    # PDE contribution
    pde_r = calculate_pde(model_, pde_points_)
    pde_loss_ = tf.reduce_mean(tf.square(pde_r))

    # Boundary contributions
    u_pred = model_(boundary_points_[0])
    payoff_loss_ = tf.reduce_mean(tf.square(boundary_values_[0] - u_pred))
    u_pred = model_(boundary_points_[1])
    side_loss_ = tf.reduce_mean(tf.square(boundary_values_[1] - u_pred))

    # Total loss
    loss_ = pde_loss_ + payoff_loss_ + side_loss_

    return loss_, pde_loss_, payoff_loss_, side_loss_


def calculate_loss_grad(model_, pde_points_, boundary_points_, boundary_values_):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model_.trainable_variables)
        loss_, pde_loss_, payoff_loss_, side_loss_ = calculate_loss(model_, pde_points_, boundary_points_,
                                                                    boundary_values_)

    g = tape.gradient(loss_, model_.trainable_variables)
    del tape

    return loss_, g, pde_loss_, payoff_loss_, side_loss_


# ############################## Evaluate with the network #################################################
# Calculate network's differentials by AD
def eval_model(model_, points_):
    tf_points_conv = tf.convert_to_tensor(points_)

    # PV and Greeks
    t, x = tf_points_conv[:, 0:1], tf_points_conv[:, 1:]
    with tf.GradientTape() as tape1:
        tape1.watch(x)  # We use this take for the 2nd order, so only need x
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([t, x])
            tf_points = tf.concat([t, x], axis=1)
            pv_ = model_(tf_points)

        delta_ = tape2.gradient(pv_, x)
        theta_ = tape2.gradient(pv_, t)

    gamma_ = tape1.batch_jacobian(delta_, x)

    # Recombine to go back to original coordinates
    fwd_ = spot * np.exp((rate - div) * t)
    spot_ = fwd_ * np.exp(x)
    delta_ = delta_.numpy()
    theta_ = theta_.numpy()
    gamma_ = gamma_.numpy()
    theta_ -= np.sum((rate - div) * delta_, axis=1, keepdims=True)

    # Recombine gamma explicitly
    for i_ in range(len(delta_)):
        for j_ in range(num_assets):
            gamma_[i_, j_, j_] -= delta_[i_, j_]
            for k in range(num_assets):
                gamma_[i_, j_, k] = gamma_[i_, j_, k] / (spot_[i_, j_] * spot_[i_, k])

    delta_ = delta_ / spot_

    return pv_.numpy(), delta_, theta_, gamma_


# ############################## Monte-Carlo ###############################################################
# Choose main and alternate direction to calculate/view Greeks
asset_idx = 0
cross_asset_idx = 1


# Direct MC simulation (non-tensor/non-AD)
def mc_simulation_pv(spot_, time_, gaussians):
    tf_spot = tf.convert_to_tensor(spot_, dtype=dtype)
    tf_time = tf.convert_to_tensor(time_, dtype=dtype)

    # Calculate deterministic forward
    fwd = tf_spot * tf.math.exp((rate - div) * tf_time)

    # Calculate final spot paths
    stdev_ = vol * tf.math.sqrt(tf_time)
    future_spot = fwd * tf.math.exp(-0.5 * stdev_ * stdev_ + stdev_ * gaussians)

    # Calculate discounted payoff
    df = tf.math.exp(-rate * tf_time)
    perf = future_spot / fixings - 1.0
    wperf = tf.reduce_min(perf, axis=1, keepdims=True)
    floored_rate = floor + tf_smooth_max(wperf, floor)
    eff_rate = cap - tf_smooth_max(-floored_rate, -cap)
    payoff_ = df * notional * eff_rate

    # Reduce
    pv_ = tf.reduce_mean(payoff_, axis=0)

    return pv_.numpy()[0]


def mc_simulation(spot_, time_, num_mc_):
    rng = np.random.RandomState(seed)
    num_assets_ = spot_.shape[0]
    means = np.zeros(shape=num_assets_)
    gaussians = rng.multivariate_normal(means, correl, size=num_mc_)

    pv_ = mc_simulation_pv(spot_, time_, gaussians)

    # Delta and Gamma
    spot_bump = 0.01 * spot_
    spot_10 = spot_.copy()
    spot_10[asset_idx] += spot_bump[asset_idx]
    pv_10 = mc_simulation_pv(spot_10, time_, gaussians)
    spot_m10 = spot_.copy()
    spot_m10[asset_idx] -= spot_bump[asset_idx]
    pv_m10 = mc_simulation_pv(spot_m10, time_, gaussians)
    spot_01 = spot_.copy()
    spot_01[cross_asset_idx] += spot_bump[cross_asset_idx]
    pv_01 = mc_simulation_pv(spot_01, time_, gaussians)
    spot_0m1 = spot_.copy()
    spot_0m1[cross_asset_idx] -= spot_bump[cross_asset_idx]
    pv_0m1 = mc_simulation_pv(spot_0m1, time_, gaussians)

    spot_11 = spot_.copy()
    spot_11[asset_idx] += spot_bump[asset_idx]
    spot_11[cross_asset_idx] += spot_bump[cross_asset_idx]
    pv_11 = mc_simulation_pv(spot_11, time_, gaussians)
    spot_1m1 = spot_.copy()
    spot_1m1[asset_idx] += spot_bump[asset_idx]
    spot_1m1[cross_asset_idx] -= spot_bump[cross_asset_idx]
    pv_1m1 = mc_simulation_pv(spot_1m1, time_, gaussians)
    spot_m11 = spot_.copy()
    spot_m11[asset_idx] -= spot_bump[asset_idx]
    spot_m11[cross_asset_idx] += spot_bump[cross_asset_idx]
    pv_m11 = mc_simulation_pv(spot_m11, time_, gaussians)
    spot_m1m1 = spot_.copy()
    spot_m1m1[asset_idx] -= spot_bump[asset_idx]
    spot_m1m1[cross_asset_idx] -= spot_bump[cross_asset_idx]
    pv_m1m1 = mc_simulation_pv(spot_m1m1, time_, gaussians)

    delta_10 = (pv_10 - pv_m10) / (2.0 * spot_bump[asset_idx])
    delta_01 = (pv_01 - pv_0m1) / (2.0 * spot_bump[cross_asset_idx])
    delta_ = np.zeros(num_assets)
    delta_[asset_idx] = delta_10
    delta_[cross_asset_idx] = delta_01

    gamma_10 = (pv_10 + pv_m10 - 2.0 * pv_) / np.power(spot_bump[asset_idx], 2.0)
    gamma_01 = (pv_01 + pv_0m1 - 2.0 * pv_) / np.power(spot_bump[cross_asset_idx], 2.0)
    gamma_11 = (pv_11 - pv_1m1 - pv_m11 + pv_m1m1) / (
                4.0 * spot_bump[asset_idx] * spot_bump[cross_asset_idx])
    gamma_ = np.zeros(shape=(num_assets, num_assets))
    gamma_[asset_idx, asset_idx] = gamma_10
    gamma_[asset_idx, cross_asset_idx] = gamma_[cross_asset_idx, asset_idx] = gamma_11
    gamma_[cross_asset_idx, cross_asset_idx] = gamma_01

    # # Theta
    time_bump = 1.0 / 365.0
    pv_1d = mc_simulation_pv(spot_, time_ - time_bump, gaussians)
    theta_ = (pv_1d - pv_) / time_bump

    return pv_, delta_, theta_, gamma_


# ############################## Comparison sample #########################################################
# Choose asset direction and number of points
num_points = 50
# If set to true, the ladder is made by points spanning the training space jointly in each dimension.
# That is, the moneyness is the same in all dimensions for a given point.
# If set to false, the ladder is only in the chosen asset direction. All other dimensions are at the
# market spot.
use_joint_ladder = False

# Generate identical points at the initial spot value
x_space = np.zeros(shape=(num_points, num_assets), dtype=dtype)

# Create a reduced spot ladder
if use_joint_ladder:
    x_space = np.linspace(xmin, xmax, num_points)
else:
    xmin_asset = xmin[asset_idx]
    xmax_asset = xmax[asset_idx]
    x_ladder = np.linspace(xmin_asset, xmax_asset, num_points)
    for i in range(num_points):
        x_space[i][asset_idx] = x_ladder[i]

# Calculate corresponding spot ladder
s_space = spot * np.exp(x_space)
s_ladder = s_space[:, asset_idx]

# ############################## Calculate MC sample #######################################################
# Calculate by Monte-Carlo
num_mc = 100000

# Loop call
mc_start = tm.time()
mc_pv = np.ndarray((num_points, 1))
mc_delta = np.ndarray((num_points, num_assets))
mc_theta = np.ndarray((num_points, 1))
mc_gamma = np.ndarray((num_points, num_assets, num_assets))
for i in range(num_points):
    pv, delta, theta, gamma = mc_simulation(s_space[i], expiry, num_mc)
    mc_pv[i] = pv
    mc_delta[i] = delta
    mc_theta[i] = theta
    mc_gamma[i] = gamma

mc_time = tm.time() - mc_start
print('Runtime(Monte-Carlo): %.1f' % mc_time + 's')


# Custom learning rate scheduler
class FlooredExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr_=1e-1, final_lr_=1e-4, decay_=0.96, decay_steps_=100):
        self.initial_lr = initial_lr_
        self.final_lr = final_lr_
        self.decay = decay_
        self.decay_steps = decay_steps_

    def __call__(self, step):
        ratio = tf.cast(step / self.decay_steps, tf.float32)
        coeff = tf.pow(self.decay, ratio)
        return self.initial_lr * coeff + self.final_lr * (1.0 - coeff)

    def get_config(self):
        config = {'initial_lr': self.initial_lr,
                  'final_lr': self.final_lr,
                  'decay': self.decay,
                  'decay_steps': self.decay_steps}
        return config


# ############################## Initialize training #######################################################
# Initialize model (one activation per hidden layer)
hidden_layers = ['softplus', 'softplus', 'softplus', 'softplus']  # 5D, 3D, 2D
# hidden_layers = ['softplus', 'softplus', 'softplus', 'softplus', 'softplus']  # 3D, 2D
num_neurons = 16  # 5D(16), 3D(8), 2D(8)

model = compose_model(hidden_layers, num_neurons)

# Leaning rate
init_lr = 1e-1  # 2D(1e-1)
final_lr = 1e-3  # 2D(1e-3)
decay = 0.97  # 2D(0.97)
steps = 100  # 2D(100)
lr_schedule = FlooredExponentialDecay(init_lr, final_lr, decay, steps)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Optimizer fields
print("Optimizer settings")
optim_fields = optimizer.get_config()
for field in optim_fields:
    print(field, ":", optim_fields[field])

# Record loss and learning rate
loss_hist = []
lr_hist = []

# Periodically save weights of the best model to date, and use this best model as a solution
# at the end of the training. The loss is used as metric to determine the best model.
best_loss_to_date = np.finfo(dtype=dtype).max
best_weight_file = "model_current_best.h5"
weights_are_saved = False

# Sample point sizes
num_final = 20 * 1000  # 5D(20,000), 4D(10,000), 3D(6,000), 2D(4,1000)
nb = 2500  # 5D(2,500), 4D(1,250), 3D(750), 2D(500)
num_pde = 20 * 1000  # 5D(20,000), 4D(10,000), 3D(6,1000), 2D(4,1000)

# Generate sample points
tf.random.set_seed(seed)
np.random.seed(seed)
pde_points, boundary_points, boundary_values = build_dataset(num_final, nb, num_pde)


# Define training step
@tf.function
def train_step(pde_points_, boundary_points_, boundary_values_):
    # Compute current loss and gradient
    loss_, grad_, pde_loss_, payoff_loss_, side_loss_ = calculate_loss_grad(model, pde_points_,
                                                                            boundary_points_,
                                                                            boundary_values_)

    # Apply gradient descent
    optimizer.apply_gradients(zip(grad_, model.trainable_variables))

    return loss_, pde_loss_, payoff_loss_, side_loss_


# ############################## Launch the training #######################################################
# Training parameters
num_epochs = 20 * 1000
redraw_dataset = False

# Prepare comparison with MS
tpoint = np.asarray([0])
points = [np.concatenate([tpoint, x]) for x in x_space]

# Start the training
t0 = tm.time()
for epoch in range(num_epochs + 1):
    # lr = optimizer.learning_rate
    lr = optimizer.learning_rate.numpy()
    # loss, pde_loss, payoff_loss, side_loss = train_step()

    loss, pde_loss, payoff_loss, side_loss = train_step(pde_points, boundary_points, boundary_values)

    loss_hist.append(loss)
    # loss_hist.append(loss.numpy())
    lr_hist.append(lr)

    # Periodically display information such as loss, learning rate, rmse to reference, etc...
    if epoch % 100 == 0:
        # md_pv = model(points)
        md_pv = model(tf.convert_to_tensor(points))
        rmse_ = np.sqrt(sk.metrics.mean_squared_error(mc_pv, md_pv))
        print('Epoch {c:4,}/{h:,}: loss={a:,.0f}, '.format(h=num_epochs, c=epoch, a=loss), end="")
        print('rmse(MC)={b:,.0f}'.format(b=rmse_), end="")
        print(', pde={b:,.0f}'.format(b=pde_loss), end="")
        print(', payoff={b:,.0f}'.format(b=payoff_loss), end="")
        print(', side={b:,.0f}'.format(b=side_loss), end="")
        print(', lr={b:.6f}'.format(b=lr))

    if redraw_dataset and epoch % 1000 == 0:
        pde_points, boundary_points, boundary_values = build_dataset(num_final, nb, num_pde)

    # Save model if best loss to date.
    if epoch % 100 == 0 and loss < best_loss_to_date:
        model.save_weights(best_weight_file)
        weights_are_saved = True
        best_loss_to_date = loss

# Reload best status
if weights_are_saved:
    print("Best model has loss={a:,.0f}".format(a=best_loss_to_date))
    model.load_weights(best_weight_file)

print("")
runtime = tm.time() - t0
print('Runtime: {:.3f} seconds'.format(runtime))

# Save the model to file
now = datetime.now()
dt_string = now.strftime("%Y%m%d-%H_%M_%S")
this_model_name = model_name + "_" + dt_string
eff_model_folder = os.path.join(model_folder, this_model_name)
print("Saving model to folder " + eff_model_folder)
model.save(eff_model_folder)

# Calculate RMSE and save model performance
md_pv, md_delta, md_theta, md_gamma = eval_model(model, points)
rmse_ = np.sqrt(sk.metrics.mean_squared_error(mc_pv, md_pv))

# Append results to stat file
stats_file = os.path.join(model_folder, model_name + "_stats.tsv")
new_dic = {'Name': [this_model_name], 'Rmse': [rmse_], 'Runtime': [runtime]}
new_df = pd.DataFrame(new_dic)
if os.path.exists(stats_file):
    stats_df = pd.read_csv(stats_file, sep="\t")
    stats_df = pd.concat([stats_df, new_df])
else:
    stats_df = new_df

# Output to csv
stats_df.to_csv(stats_file, sep="\t", index=False)
