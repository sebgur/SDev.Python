# Import relevant modules
import numpy as np
import tensorflow as tf
import scipy.stats
import sklearn as sk
from platform import python_version
import time as tm
import matplotlib.pyplot as plt
import math
import warnings
from pysabr import hagan_2002_lognormal_sabr as sabr
from pysabr import black

print("Python version: " + python_version())
print("TensorFlow version: " + tf.__version__)
print("TF-Keras version: " + tf.keras.__version__)
print("NumPy version: " + np.__version__)
print("SciPy version: " + scipy.__version__)
print("SciKit version: " + sk.__version__)

# Global settings
shift = 0.03  # To account for negative rates in the SABR model
beta = 0.5  # Fixed to avoid the parameter redundancy
dtype = 'float32'
np.seterr(all='warn')
warnings.filterwarnings('error')

# Helpers
def price_function(expiry, fwd, strike, alpha, nu, rho):
    vol = sabr.lognormal_vol(strike + shift, fwd + shift, expiry, alpha, beta, rho, nu)
    r = 0.0  # Use forward prices (0 discount rate)
    fwd_price = black.lognormal_call(strike + shift, fwd + shift, expiry, vol, r)
    return fwd_price

from scipy.stats import norm
from scipy.optimize import minimize, minimize_scalar

def bachelier_price(expiry, fwd, strike, vol):
    stdev = vol * expiry**0.5
    d = (fwd - strike) / stdev
    price = stdev * (d * norm.cdf(d) + norm.pdf(d))
    return price

# Direct method by numerical inversion using Brent
def price_to_nvol(expiry, fwd, strike, price):
    options = {'xtol': 1e-4, 'maxiter': 100, 'disp': False}
    xmin = 1e-6
    xmax = 1.0

    def error(vol):
        premium = bachelier_price(expiry, fwd, strike, vol)
        return (premium - price) ** 2

    res = minimize_scalar(fun=error, bracket=(xmin, xmax), options=options, method='brent')

    return res.x

# P. Jaeckel's method in "Implied Normal Volatility", 6th Jun. 2017
def price_to_nvol_jaeckel(expiry, fwd, strike, price, is_call=True):
    # Special case at ATM
    if np.abs(fwd - strike) < 1e-8:
        return price * np.sqrt(2.0 * np.pi) / np.sqrt(expiry)

    # General case
    tilde_phi_star_C = -0.001882039271
    theta = 1.0 if is_call else -1.0

    tilde_phi_star = -np.abs(price - np.maximum(theta * (fwd - strike), 0.0) ) / np.abs(fwd - strike)
    em5 = 1e-5

    if tilde_phi_star < tilde_phi_star_C:
        g = 1.0 / (tilde_phi_star - 0.5)
        g2 = g**2
        em3 = 1e-3
        num = 0.032114372355 - g2 * (0.016969777977 - g2 * (2.6207332461 * em3 - 9.6066952861 * em5 * g2))
        den = 1.0 - g2 * (0.6635646938 - g2 * (0.14528712196 - 0.010472855461 * g2))
        eta_bar = num / den
        xb = g * (eta_bar * g2 + 1.0 / np.sqrt(2.0 * np.pi))
    else:
        h = np.sqrt(-np.log(-tilde_phi_star))
        num = 9.4883409779 - h * (9.6320903635 - h * (0.58556997323 + 2.1464093351 * h))
        den = 1.0 - h * (0.65174820867 + h * (1.5120247828 + 6.6437847132 * em5 * h))
        xb = num / den

    q =  (norm.cdf(xb) + norm.pdf(xb) / xb - tilde_phi_star) / norm.pdf(xb)
    xb2 = xb**2
    num = 3.0 * q * xb2 * (2.0 - q * xb * (2.0 + xb2))
    den = 6.0 + q * xb * (-12.0 + xb * (6.0 * q + xb * (-6.0 + q * xb * (3.0 + xb2))))
    xs = xb + num / den
    sigma = np.abs(fwd - strike) / (np.abs(xs) * np.sqrt(expiry))
    return sigma

from sklearn.preprocessing import StandardScaler

# LearningModel class, which incorporates the Keras model and its scalers in a single interface
class LearningModel:
    def __init__(self, model):
        self.model = model
        self.x_scaler = StandardScaler(copy=True)
        self.y_scaler = StandardScaler(copy=True)
        self.is_scaled = False
        # self.epochs = []
        # self.losses = []
        # self.accuracies = []
        # self.offset = 0

    def train(self, x_set, y_set, epochs, batch_size):  #, call_back):
        if not self.is_scaled:  # Then we scale the inputs and outputs
            self.x_scaler.fit(x_set)
            self.y_scaler.fit(y_set)
            # call_back.set_scalers(self.x_scaler, self.y_scaler)
            self.is_scaled = True

        x_scaled = self.x_scaler.transform(x_set)
        y_scaled = self.y_scaler.transform(y_set)

        history = self.model.fit(x_scaled, y_scaled, epochs=epochs, batch_size=batch_size, shuffle=True,
                                 verbose=1)#, callbacks=[call_back])

        return history

        # epochs, accuracies = call_back.convergence()

        # self.losses.extend(history.history['loss'])
        # self.accuracies.extend(accuracies)

        # num_epochs = len(epochs)
        # for i in range(num_epochs):
        #     self.epochs.append(self.offset + epochs[i])

        # self.offset += num_epochs

    def predict(self, x_test):
        x_scaled = self.x_scaler.transform(x_test)
        y_scaled = self.model(x_scaled)
        y_test = self.y_scaler.inverse_transform(y_scaled)
        return y_test

    # def convergence(self):
    #     return self.epochs, self.losses, self.accuracies

    # def clear_training(self):
    #     self.epochs.clear()
    #     self.losses.clear()
    #     self.accuracies.clear()
    #     self.offset = 0


# Custom learning rate scheduler, exponentially decreases between given values
class FlooredExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-1, final_lr=1e-4, decay=0.96, decay_steps=100):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay = decay
        self.decay_steps = decay_steps

    def __call__(self, step):
        ratio = tf.cast(step / self.decay_steps, tf.float32)
        coeff = tf.pow(self.decay, ratio)
        return self.initial_lr * coeff + self.final_lr * (1.0 - coeff)

    def get_config(self):
        config = { 'initial_lr': self.initial_lr,
                   'final_lr': self.final_lr,
                   'decay': self.decay,
                   'decay_steps': self.decay_steps }
        return config


# Adding layers
kinits = tf.keras.initializers
klayers = tf.keras.layers

def add_hidden_layer(model_, neurons_, activation_):
    init = kinits.glorot_normal

    model_.add(klayers.Dense(neurons_, activation=activation_, kernel_initializer=init,
                             use_bias=True, bias_initializer=kinits.Constant(0.1)))
    

# Set up the network architecture
def compose_model(dimension_, hidden_layers_, num_neurons_, dropout_=0.2):
    model_ = tf.keras.Sequential()

    # multi-d inputs
    model_.add(tf.keras.Input(dimension_))

    # Hidden layers
    for hl in hidden_layers_:
        add_hidden_layer(model_, num_neurons_, hl)
        model_.add(klayers.Dropout(dropout_))

    # 1d outputs
    model_.add(klayers.Dense(1))

    return model_


num_samples = 100000

rngSeed = 42
rng = np.random.RandomState(rngSeed)

t0 = tm.time()
# Generate inputs
expiry = rng.uniform(1.0 / 12.0, 5.0, (num_samples, 1))
fwd = rng.uniform(-0.01, 0.04, (num_samples, 1))
ln_vol = rng.uniform(0.05, 0.25, (num_samples, 1))  # Specify log-normal vol
alpha = ln_vol * np.power(np.abs(fwd + shift), 1 - beta)  # Rough order of magnitude for alpha
nu = rng.uniform(0.20, 0.80, (num_samples, 1))
rho = rng.uniform(-0.40, 0.40, (num_samples, 1))
spread = rng.uniform(-300, 300, (num_samples, 1))  # Specify spreads in bps
strike = fwd + spread / 10000
print('Generate inputs: {:.3f} seconds'.format(tm.time() - t0))

# Generate outputs
t0 = tm.time()
fv = np.ndarray((num_samples, 1))
for i_ in range(num_samples):
    fv[i_, 0] = price_function(expiry[i_, 0], fwd[i_, 0], strike[i_, 0], alpha[i_, 0], nu[i_, 0], rho[i_, 0])

print('Generate prices: {:.3f} seconds'.format(tm.time() - t0))

t0 = tm.time()
normal_vol = []
for i_ in range(num_samples):
    try:
        normal_vol.append(price_to_nvol_jaeckel(expiry[i_, 0], fwd[i_, 0], strike[i_, 0], fv[i_, 0]))
    except Exception:
        normal_vol.append(-12345.6789)

print('Inversion: {:.3f} seconds'.format(tm.time() - t0))


# Create input set
x_set_raw = np.column_stack((expiry, fwd, strike, alpha, nu, rho))

# Filter out bad data and create output set
t0 = tm.time()
min_vol = 0.0001
max_vol = 0.1

x_set = []
y_set = []
for i in range(num_samples):
    nvol = normal_vol[i]
    if math.isnan(nvol) == False and nvol > min_vol and nvol < max_vol:
        x_set.append(x_set_raw[i])
        y_set.append(nvol)

print('Cleansing: {:.3f} seconds'.format(tm.time() - t0))

# Reshape
x_set = np.asarray(x_set)
y_set = np.asarray(y_set)
y_set = y_set.reshape((y_set.shape[0], 1))

# Display information
in_dimension = x_set.shape[1]
out_dimension = y_set.shape[1]
print("Input dimension: " + str(in_dimension))
print("Output dimension: " + str(out_dimension))
print("Dataset size: " + str(x_set.shape[0]))

# Tested activations: tanh, softplus, softmax, relu, selu, elu, sigmoid
hidden_layers = ['softplus', 'softplus', 'softplus']
num_neurons = 16

# Create model
dropout = 0.00
keras_model = compose_model(in_dimension, hidden_layers, num_neurons, dropout)

# Learning rate scheduler
init_lr = 1e-1
final_lr = 1e-4
decay = 0.97
steps = 100
lr_schedule = FlooredExponentialDecay(init_lr, final_lr, decay, steps)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile
keras_model.compile(loss='mse', optimizer=optimizer)
model = LearningModel(keras_model)

# Display optimizer fields
print("Optimizer settings")
optim_fields = optimizer.get_config()
for field in optim_fields:
    print(field, ":", optim_fields[field])

# Train the network
epochs = 1000
batch_size = 1000
history = model.train(x_set, y_set, epochs, batch_size)
