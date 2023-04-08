import numpy as np
import tensorflow as tf
import scipy.stats
import sklearn as sk
from platform import python_version
import time as tm
# import matplotlib.pyplot as plt
import math
import warnings
from pysabr import hagan_2002_lognormal_sabr as sabr
from pysabr import black
from machinelearning.FlooredExponentialDecay import FlooredExponentialDecay
from machinelearning.topology import compose_model
from machinelearning.LearningModel import LearningModel
from analytics.bachelier import impliedvol

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
        normal_vol.append(impliedvol(expiry[i_, 0], fwd[i_, 0], strike[i_, 0], fv[i_, 0]))
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

# Initialize the model
hidden_layers = ['softplus', 'softplus', 'softplus']
num_neurons = 16
dropout = 0.00
keras_model = compose_model(in_dimension, 1, hidden_layers, num_neurons, dropout)

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
