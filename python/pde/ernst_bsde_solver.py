from tools.settings import apply_settings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ernst_BsdeModel import BsdeModel
# from ernst_BsdeModel import MyClass
from time import time
# from mpl_toolkits.mplot3d import Axes3D

apply_settings()

# ############################ Runtime configuration ##################################################################
print("<><><><> Setting up runtime configuration <><><><>")
debug = False

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
print('TensorFlow version used: {}'.format(tf.__version__))

# Final time
T = 1.

# Spatial dimensions
dim = 100

# Number of equidistant intervals in time
N = 20

# Derive time step size and t_space
dt = T/N
t_space = np.linspace(0, T, N + 1)
if debug:
    print("<><><><> Time space discretization <><><><>")
    print(t_space)

# Point-of-interest at t=0
x = np.zeros(dim)

# Diffusive term is assumed to be constant
sigma = np.sqrt(2)


def draw_x_and_dw(num_samples_, x_):
    """ Draw num_sample paths of stochastic process X and increments of Brownian motions dW """

    dim_ = x_.shape[0]

    # Draw all increments of W at once
    dw = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(num_samples_, dim_, N)).astype(DTYPE)

    # Initialize the array X
    x_vec = np.zeros((num_samples_, dim_, N + 1), dtype=DTYPE)

    # Set starting point to x for each draw
    x_vec[:, :, 0] = np.ones((num_samples_, dim_)) * x_

    for i_ in range(N):
        # This corresponds to the Euler-Maruyama Scheme
        x_vec[:, :, i_ + 1] = x_vec[:, :, i_] + sigma * dw[:, :, i_]

    # Return simulated paths as well as increments of Brownian motion
    return x_vec, dw


if debug:
    num_samples = 3
    print("Displaying " + str(num_samples) + " Monte-Carlo paths")
    x_path, dw_path = draw_x_and_dw(num_samples, np.zeros(1))
    print(x_path)
    # Plot
    fix, ax = plt.subplots(1)
    for ii in range(num_samples):
        ax.plot(t_space, x_path[ii, 0, :])
    ax.set_xlabel("$t$")
    ax.set_ylabel("$X_t")
    plt.show()

###################################################################################################
# ######## 3. Simulation of \hat{Y}
###################################################################################################


def simulate_y(inp, model_, fun_f_):
    """ Perform the forward sweep through the network.
    Inputs:
        inp - (X, dW)
        model - model of neural network, contains
            - u0  - variable approximating u(0, x)
            - gradu0 - variable approximating nabla u(0, x)
            - gradui - list of NNs approximating the mapping: x -> nabla u(t_i, x)
        fun_f - function handle for cost function f
    """

    x_, dw_ = inp
    num_sample = X.shape[0]

    e_num_sample = tf.ones(shape=[num_sample, 1], dtype=DTYPE)

    # Value approximation at t0
    y_ = e_num_sample * model_.u0

    # Gradient approximation at t0
    z = e_num_sample * model_.gradu0

    for i_ in range(N - 1):
        t = t_space[i_]

        # Determine terms in right-hand side of Y-update at t_i
        eta1 = - fun_f_(t, x_[:, :, i_], y_, z) * dt
        eta2 = tf.reduce_sum(z * dw_[:, :, i_], axis=1, keepdims=True)

        # Compute new value approximations at t_{i+1}
        y_ = y_ + eta1 + eta2

        # Obtain gradient approximations at t_{i+1}
        # Scaling the variable z by 1/dim improves the convergence properties
        # and has been used in the original code https://github.com/frankhan91/DeepBSDE
        # z still approximates \sigma^T \nabla u, but the network learns to represent
        # a scaled version.
        z = model_.gradui[i_](x_[:, :, i_ + 1]) / dim

    # Final step
    eta1 = - fun_f_(t_space[N - 1], x_[:, :, N - 1], y_, z) * dt
    eta2 = tf.reduce_sum(z * dw_[:, :, N - 1], axis=1, keepdims=True)
    y_ = y_ + eta1 + eta2

    return y_
###################################################################################################
# ######## 4. Evaluation of loss function
###################################################################################################


def loss_fn(inp, model_, fun_f_, fun_g_):
    """ Compute the mean-squarred error of the difference of Y_T and g(X_T)
    Inputs:
        inp - (X, dW)
        model - model of neural network containing u0, gradu0, gradui
        fun_f - function handle for cost function f
        fun_g - function handle for terminal condition g
    """
    x_, _ = inp

    # Forward pass to compute value estimates
    y_pred = simulate_y(inp, model_, fun_f_)

    # Final time condition, i.e., evaluate g(X_T)
    y_ = fun_g_(x_[:, :, -1])

    # Compute mean squared error
    y_diff = y_ - y_pred
    loss_ = tf.reduce_mean(tf.square(y_diff))

    return loss_


###################################################################################################
# ######## 5. Computation of gradient w.r.t. network parameters
###################################################################################################

@tf.function
def compute_grad(inp, model_, fun_f_, fun_g_):
    """ Computes the gradient of the loss function w.r.t. the trainable variables theta.
    Inputs:
        inp - (X, dW)
        model - model of neural network containing u0, gradu0, gradui
        fun_f - function handle for cost function f
        fun_g - function handle for terminal condition g
    """

    with tf.GradientTape() as tape:
        loss_ = loss_fn(inp, model_, fun_f_, fun_g_)

    grad_ = tape.gradient(loss_, model_.trainable_variables)

    return loss_, grad_

###################################################################################################
# ######## Example: Quadratic Gaussian control problem
###################################################################################################


# Define cost function f, remember that z approximates \sigma^T \nabla u
def fun_f(t_, x_, y_, z_):
    return - tf.reduce_sum(tf.square(z_), axis=1, keepdims=True) / (sigma**2)


# Set terminal value function g
def fun_g(x_):
    return tf.math.log((1+tf.reduce_sum(tf.square(x_), axis=1, keepdims=True)) / 2)


# Set learning rate
lr = 1e-2
# Choose optimizer for gradient descent step
optimizer = tf.keras.optimizers.Adam(lr, epsilon=1e-8)

# Initialize neural network architecture
model = BsdeModel(dim=dim, num_steps=N)
y_star = 4.59016

# Initialize list containing history of losses
history = []

t0 = time()

num_epochs = 2000

# Initialize header of output
print('  Iter        Loss        y   L1_rel    L1_abs   |   Time  Stepsize')

for i in range(num_epochs):

    # Each epoch we draw a batch of 64 random paths
    X, dW = draw_x_and_dw(64, x)

    # Compute the loss as well as the gradient
    loss, grad = compute_grad((X, dW), model, fun_f, fun_g)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    # Get current Y_0 \approx u(0,x)
    y = model.u0.numpy()[0]

    currtime = time() - t0
    l1abs = np.abs(y - y_star)
    l1rel = l1abs / y_star

    hentry = (i, loss.numpy(), y, l1rel, l1abs, currtime, lr)
    history.append(hentry)
    if i % 10 == 0:
        print('{:5d} {:12.4f} {:8.4f} {:8.4f}  {:8.4f}   | {:6.1f}  {:6.2e}'.format(*hentry))

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
xrange = range(len(history))
ax[0].semilogy(xrange, [e[1] for e in history], 'k-')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('training loss')
ax[1].plot(xrange, [e[2] for e in history])
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('$u(0,x)$')
plt.show()
