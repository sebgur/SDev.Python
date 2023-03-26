import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
import numpy as np
from tools.black import black_formula

# Strongly inspired by Blechschmidt and Ernst
# https://onlinelibrary.wiley.com/doi/full/10.1002/gamm.202100006

# ######## ToDo ###################################################################################
# Switch to Black-Scholes
# Calculate closed-form
# Calculate loss to closed-form
# Display


###################################################################################################
# ######## Initialize problem
###################################################################################################
# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
print('TensorFlow version used: {}'.format(tf.__version__))

# Set constants
expiry = 1.25
repo_rate = 0.01
disc_rate = 0.015
div_rate = 0.005
vol = 0.20
strike = 100.0

# Final time
T = tf.constant(expiry, dtype=DTYPE)

# Spatial dimension
dim = 1

# Spatial domain of interest at t=0 (hyperrectangle)
# spot_min = tf.zeros(dim, dtype=DTYPE)
# spot_max = tf.ones(dim, dtype=DTYPE)
spot_min = 90 * tf.ones(dim, dtype=DTYPE)
spot_max = 110 * tf.ones(dim, dtype=DTYPE)

# Interest rate
r = tf.constant(disc_rate, dtype=DTYPE)

# Drift
mu = tf.constant(repo_rate - div_rate, dtype=DTYPE)

# Strike
K = tf.constant(strike, dtype=DTYPE)

# Diffusion coefficient is assumed to be constant
# sigma = np.sqrt(2, dtype=DTYPE)
sigma = tf.constant(vol, dtype=DTYPE)


# Define payoff at maturity
def disc_payoff(x_):
    return tf.exp(-r * T) * tf.maximum(tf.math.reduce_prod(x_, axis=1, keepdims=True) - K, 0.0)
    # return tf.reduce_sum(tf.pow(x_, 2), axis=1, keepdims=True)


# Define exact reference solution
def bs_value(t, x_):
    df = np.exp(-r * T)
    bs_price = 0
    return df * bs_price


# Define exact reference solution
def ref_value(t, x_):
    return tf.reduce_sum(tf.pow(x_, 2), axis=1, keepdims=True) + 2 * t * dim
    # return tf.reduce_sum(tf.pow(x_, 2), axis=1, keepdims=True) + 2 * t * dim


###################################################################################################
# ######## Generation of training data
###################################################################################################
@tf.function
def calc_underlying_paths(num_samples):
    # Draw initial spots
    dim_ = spot_min.shape[0]

    x0_ = spot_min + tf.random.uniform((num_samples, dim_), dtype=DTYPE) * (spot_max - spot_min)

    # Draw final spots along the paths
    zi = tf.random.normal(shape=(num_samples, dim_), dtype=DTYPE)
    # xt = x0_ + sigma * np.sqrt(T) * zi
    xt = x0_ * tf.exp((mu - tf.square(sigma) / 2.0) * T + sigma * tf.sqrt(T) * zi)

    # Return simulated paths as well as increments of Brownian motion
    return tf.stack([x0_, xt], 2)


###################################################################################################
# ######## Set up neural network model
###################################################################################################
def init_model(dim_, activation='tanh', num_hidden_neurons=100, num_hidden_layers=2,
               initializer=tf.keras.initializers.GlorotUniform()):

    model_ = tf.keras.Sequential()
    model_.add(tf.keras.layers.Input(dim_))
    model_.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))

    for _ in range(num_hidden_layers):
        model_.add(tf.keras.layers.Dense(num_hidden_neurons, activation=None, use_bias=False,
                                         kernel_initializer=initializer))
        model_.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))
        model_.add(tf.keras.layers.Activation(activation))

    model_.add(tf.keras.layers.Dense(1, activation=None, use_bias=False, kernel_initializer=initializer))
    model_.add(tf.keras.layers.BatchNormalization(epsilon=1e-6))

    return model_


###################################################################################################
# ######## Define functions for the objective and its gradient
###################################################################################################
def loss_fn(x_, y_, model_, training=False):
    """ Compute the MSE between the current model prediction model(X) and the values y.
    Inputs:
        X - approximation to the state process X
        y - target value
        model - model of neural network approximating x -> u(T,x)
        training - boolean flag to indicate training """
    x0_ = x_[:, :, 0]
    y_pred = model_(x0_, training)

    # Return mean squared error
    return tf.reduce_mean(tf.math.squared_difference(y_, y_pred))


@tf.function
def compute_grad(x_, y_, model_, training=False):
    """ Compute the gradient of the loss function w.r.t. the trainable variables theta.
        Inputs:
            X - approximation to the state process X
            y - target value
            model - model of neural network approximating x -> u(T,x)
            training - boolean flag to indicate training """
    with tf.GradientTape() as tape:
        loss_ = loss_fn(x_, y_, model_, training)

    grad_ = tape.gradient(loss_, model_.trainable_variables)

    return loss_, grad_


###################################################################################################
# ######## Solve the PDE
###################################################################################################
# Original experiment
model = init_model(dim_=dim, num_hidden_neurons=200)
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([250001, 500001], [1e-3, 1e-4, 1e-5])
suffix = 'orig'

# Set number of training epochs
num_epochs = 10000  # 750001

# Set batch size
batch_size = 8192

# Choose an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-8)

# Initialize list containing history of losses
hist_loss = []
error_hist = []

# Randomly choose a test set from D to approximate errors by means of Monte-Carlo sampling
n_test = 1000
x = calc_underlying_paths(n_test)
x_test = x[:, :, 0]
# Compute exact solution
y_test = ref_value(T, x_test)


# Define a training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Draw batch of random paths
    x_ = calc_underlying_paths(batch_size)

    # Evaluate g at x_T
    y_ = disc_payoff(x_[:, :, -1])

    # And compute the loss as well as the gradient
    loss_, grad = compute_grad(x_, y_, model, training=True)

    # Perform gradient step
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return loss_


# Start timer
t0 = time()

# Set interval to estimate errors
log_interval = 1000

# Initialize header of output
print('  Iter        Loss   L1_rel   L2_rel   Linf_rel |   L1_abs   L2_abs   Linf_abs  |    Time  Stepsize')

# Loop to train model
print("Training...")
for i in range(num_epochs):
    # Perform training step
    loss = train_step()
    hist_loss.append(loss)

    if i % log_interval == 0:
        # Compute current prediction on test set
        Ypred = model(x_test, training=False)

        # Compute absolute and relative errors
        abs_error = np.abs(Ypred - y_test)
        rel_error = abs_error / y_test
        L2_rel = tf.sqrt(tf.reduce_mean(tf.pow(rel_error, 2))).numpy()
        L1_rel = tf.reduce_mean(tf.abs(rel_error)).numpy()
        Linf_rel = tf.reduce_max(tf.abs(rel_error)).numpy()

        L2_abs = tf.sqrt(tf.reduce_mean(tf.pow(abs_error, 2))).numpy()
        L1_abs = tf.reduce_mean(tf.abs(abs_error)).numpy()
        Linf_abs = tf.reduce_max(tf.abs(abs_error)).numpy()
        total_time = time() - t0
        stepsize = optimizer.lr(optimizer.iterations).numpy()
        err = (i, loss.numpy(), L1_rel, L2_rel, Linf_rel, L1_abs, L2_abs, Linf_abs, total_time, stepsize)
        error_hist.append(err)
        print('{:5d} {:12.4f} {:8.4f} {:8.4f}   {:8.4f} | {:8.4f} {:8.4f} {:8.4f}  |  {:6.1f}  {:6.2e}'.format(*err))

print(time() - t0)

# Plot training history
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.semilogy(range(len(hist_loss)), hist_loss, 'k-')
ax.set_xlabel('epoch')
ax.set_ylabel('training loss')
plt.show()
