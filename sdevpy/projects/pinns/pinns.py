import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tools.black import black_formula
from time import time
from enum import Enum
import scipy.stats


class PayoffType(Enum):
    Forward = 1
    Call = 2
    Put = 3


###################################################################################################
# ######## Runtime configuration
###################################################################################################
show_colocation = False
show_pv = True
show_loss = True
n_epochs = 200
DTYPE = 'float32'

# Set number of data points
n_final = 250  # 50
nb = 25  # 25
n_boundary = 2 * nb  # 50  # 50
n_pde = 10000  # 10000

###################################################################################################
# ######## Set problem specific data
###################################################################################################
# Set data type
tf.keras.backend.set_floatx(DTYPE)

# Set constants
vol = 0.25
drift = 0.0
rate = 0.0
T = 10.0
S0 = 100.0
F = S0
k = 0.95 * S0
K = tf.constant(k, dtype=DTYPE)
scale = tf.constant(S0, dtype=DTYPE)
floor = tf.constant(0.0, dtype=DTYPE)
# payoff_type = PayoffType['Forward']
# payoff_type = PayoffType['Call']
payoff_type = PayoffType['Put']
is_call = True if payoff_type == PayoffType.Call else False

stdev = vol * np.sqrt(T)
conf = 0.99
p = scipy.stats.norm.ppf(conf)
xmin = F * np.exp(-0.5 * stdev * stdev - p * stdev)
xmax = F * np.exp(-0.5 * stdev * stdev + p * stdev)
print(xmin)
print(xmax)

# Boundaries
tmin = T
tmax = 0.0
# xmin = 0.8 * S0
# xmax = 1.2 * S0


# Define final payoff
def final_payoff(x):
    if payoff_type == PayoffType.Forward:
        return (x - K) / scale
    elif payoff_type == PayoffType.Call:
        return tf.math.maximum(x - K, floor) / scale
    elif payoff_type == PayoffType.Put:
        return tf.math.maximum(K - x, floor) / scale
    else:
        raise Exception("Unknown payoff type: " + str(payoff_type.value))


# Define boundary condition
def lw_boundary(t, x):
    n = x.shape[0]
    if payoff_type == PayoffType.Forward:
        return tf.ones((n, 1), dtype=DTYPE) * (x - K) / scale
    elif payoff_type == PayoffType.Call:
        return tf.zeros((n, 1), dtype=DTYPE)
    elif payoff_type == PayoffType.Put:
        return tf.ones((n, 1), dtype=DTYPE) * (K - x) / scale
    else:
        raise Exception("Unknown payoff type: " + str(payoff_type.value))


def up_boundary(t, x):
    n = x.shape[0]
    if payoff_type == PayoffType.Forward:
        return tf.ones((n, 1), dtype=DTYPE) * (x - K) / scale
    elif payoff_type == PayoffType.Call:
        return tf.ones((n, 1), dtype=DTYPE) * (x - K) / scale
    elif payoff_type == PayoffType.Put:
        return tf.zeros((n, 1), dtype=DTYPE)
    else:
        raise Exception("Unknown payoff type: " + str(payoff_type.value))


# PDE = 0
def pde(t, x, u, u_t, u_x, u_xx):
    return u_t + 0.5 * vol * vol * x * x * u_xx


###################################################################################################
# ######## Generate a set of collocation points
###################################################################################################
# Lower bounds
lb = tf.constant([tmin, xmin], dtype=DTYPE)

# Upper bounds
ub = tf.constant([tmax, xmax], dtype=DTYPE)

# Set random seed for reproducible results
tf.random.set_seed(0)

# Draw final payoff samples
t_0 = tf.ones((n_final, 1), dtype=DTYPE) * lb[0]
x_0 = tf.random.uniform((n_final, 1), lb[1], ub[1], dtype=DTYPE)
payoffPoints = tf.concat([t_0, x_0], axis=1)
# Evaluate payoff
payoffValues = final_payoff(x_0)

# Draw boundary samples
lw_t_b = tf.random.uniform((nb, 1), lb[0], ub[0], dtype=DTYPE)
lw_x_b = tf.ones((nb, 1), dtype=DTYPE) * lb[1]
# Reuse the same times
up_t_b = lw_t_b
up_x_b = tf.ones((nb, 1), dtype=DTYPE) * ub[1]
# Concatenate
t_b = tf.concat([lw_t_b, up_t_b], axis=0)
x_b = tf.concat([lw_x_b, up_x_b], axis=0)
boundary_points = tf.concat([t_b, x_b], axis=1)

# Evaluate boundary condition at (t_b, x_b)
lwBoundaryValues = lw_boundary(lw_t_b, lw_x_b)
upBoundaryValues = up_boundary(up_t_b, up_x_b)
boundary_values = tf.concat([lwBoundaryValues, upBoundaryValues], axis=0)

# Draw uniformly sampled collocation points
t_r = tf.random.uniform((n_pde, 1), lb[0], ub[0], dtype=DTYPE)
x_r = tf.random.uniform((n_pde, 1), lb[1], ub[1], dtype=DTYPE)
pde_points = tf.concat([t_r, x_r], axis=1)

# Collect boundary and initial data in lists
other_points = [payoffPoints, boundary_points]
other_values = [payoffValues, boundary_values]

###################################################################################################
# ######## Illustrate collocation points
###################################################################################################
if show_colocation:
    fig = plt.figure(figsize=(9, 6))
    plt.scatter(t_0, x_0, c=payoffValues, marker='X', vmin=-1, vmax=1)
    plt.scatter(t_b, x_b, c=boundary_values, marker='X', vmin=-1, vmax=1)
    plt.scatter(t_r, x_r, c='r', marker='.', alpha=0.1)
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Positions of collocation points and boundary date')
    plt.show()


###################################################################################################
# ######## Set up network architecture
###################################################################################################
def init_model(num_hidden_layers=8, num_neurons_per_layer=20):
    # Initialize a feedforward neural network
    model_ = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    model_.add(tf.keras.Input(2))

    # Introduce a scaling layer to map input to [lb, ub]
    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0)
    model_.add(scaling_layer)

    # Append hidden layers
    for _ in range(num_hidden_layers):
        model_.add(tf.keras.layers.Dense(num_neurons_per_layer,
                                         activation=tf.keras.activations.get('tanh'),
                                         kernel_initializer='glorot_normal'))

    # Output is one-dimensional
    model_.add(tf.keras.layers.Dense(1))

    return model_


###################################################################################################
# ######## Define routines to determine loss and gradient
###################################################################################################
def get_r(model_, pde_points_):
    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        t, x = pde_points_[:, 0:1], pde_points_[:, 1:2]

        # Variables t and x are watched during tape to compute derivatives u_t and u_x
        tape.watch(t)
        tape.watch(x)

        # Determine residual
        u = model_(tf.stack([t[:, 0], x[:, 0]], axis=1))

        # Compute gradient u_x within the GradientTape since we need the second derivatives
        u_x = tape.gradient(u, x)

    u_t = tape.gradient(u, t)
    u_xx = tape.gradient(u_x, x)

    del tape

    return pde(t, x, u, u_t, u_x, u_xx)


def compute_loss(model_, pde_points_, other_points_, other_values_):
    # Compute phi^r
    r = get_r(model_, pde_points_)
    phi_r = tf.reduce_mean(tf.square(r))

    # Initialize loss
    loss_ = phi_r

    # Add phi^0 and phi^b to the loss
    for i_ in range(len(other_points_)):
        u_pred = model_(other_points_[i_])
        loss_ += tf.reduce_mean(tf.square(other_values_[i_] - u_pred))

    return loss_


def get_grad(model_, pde_points_, other_points_, other_values_):
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with respect to trainable variables
        tape.watch(model_.trainable_variables)
        loss_ = compute_loss(model_, pde_points_, other_points_, other_values_)

    g = tape.gradient(loss_, model_.trainable_variables)
    del tape

    return loss_, g


###################################################################################################
# ######## Set up optimizer and define training steps
###################################################################################################
# Initialize model aka u_\theta
model = init_model()

# We choose a piecewise decay of the learning rate, i.e. the step size in the gradient descent
# type algorithm the first 1000 steps use a learning rate of 0.01,
# from 1000 - 3000: learning rate = 0.001
# from 3000 onwards: learning rate = 0.0005
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 5e-4])

# Choose the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=lr)


# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss_, grad_theta = get_grad(model, pde_points, other_points, other_values)

    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss_


###################################################################################################
# ############# Training
###################################################################################################
hist = []
t0 = time()

for i in range(n_epochs + 1):
    loss = train_step()

    # Append current loss to hist
    hist.append(loss.numpy())

    # Output current loss after 50 iterates
    if i % 50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i, loss))

# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))


###################################################################################################
# ######## Plot solution and evolution of loss
###################################################################################################
# #### Plot 1d solution
def plot_slice(n_points_, t_, x_min, x_max):
    t_space_ = t_
    x_space_ = np.linspace(x_min, x_max, n_points_)
    t_mat, x_mat = np.meshgrid(t_space_, x_space_)
    x_pred = np.vstack([t_mat.flatten(), x_mat.flatten()]).T
    y_pred = model(tf.cast(x_pred, DTYPE)) * scale
    closed_form = black_formula(x_space_, k, vol, T - t_, is_call)

    title = str(payoff_type.name) + " along spot direction at T = "
    plt.title(title + '%.2f' % t_ + ', K = %.2f' % k)
    plt.xlabel('Spot')
    plt.ylabel('PV')
    plt.plot(x_space_, closed_form, color='blue', label='Closed-Form')
    plt.plot(x_space_, y_pred, color='red', label='NN')
    plt.legend(loc='upper right')


if show_pv:
    n_points = 100
    # Plot solution
    plt.ioff()
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.40)
    plt.subplot(3, 2, 1)
    plot_slice(n_points, T / 100, lb[1], ub[1])
    plt.subplot(3, 2, 2)
    plot_slice(n_points, T / 10, lb[1], ub[1])
    plt.subplot(3, 2, 3)
    plot_slice(n_points, T / 4, lb[1], ub[1])
    plt.subplot(3, 2, 4)
    plot_slice(n_points, T / 2, lb[1], ub[1])
    plt.subplot(3, 2, 5)
    plot_slice(n_points, 3 * T / 4, lb[1], ub[1])
    plt.subplot(3, 2, 6)
    plot_slice(n_points, 9 * T / 10, lb[1], ub[1])
    plt.show()

# Plot the evolution of loss
if show_loss:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.semilogy(range(len(hist)), hist, 'k-')
    ax.set_xlabel('$n_{epoch}$')
    ax.set_ylabel('$\\phi_{n_epoch}}$')
    plt.show()
