""" Check why xsabr_fit doesn't work as well as colab """
import os
import time as tm
import math
import warnings
from platform import python_version
import numpy as np
# import pandas as pd
import tensorflow as tf
import scipy.stats
import sklearn as sk
from HaganSabrGenerator import HaganSabrGenerator, ShiftedHaganSabrGenerator
import settings
from tools import FileManager
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


# ############## Runtime configuration ############################################################
data_folder = os.path.join(settings.WORKFOLDER, "XSABRSamples")
print("Looking for data folder " + data_folder)
FileManager.check_directory(data_folder)
# model_type = "HaganSABR"
MODEL_TYPE = "ShiftedHaganSABR"
data_file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")

# Generator factory
if MODEL_TYPE == "HaganSABR":
    generator = HaganSabrGenerator()
elif MODEL_TYPE == "ShiftedHaganSABR":
    generator = ShiftedHaganSabrGenerator()
else:
    raise ValueError("Unknown model: " + MODEL_TYPE)

# Global settings
SHIFT = 0.03  # To account for negative rates in the SABR model
BETA = 0.5  # Fixed to avoid the parameter redundancy
DTYPE = 'float32'
np.seterr(all='warn')
warnings.filterwarnings('error')


# Generate samples
NUM_SAMPLES = 100000
SEED = 42
rng = np.random.RandomState(SEED)

t0 = tm.time()
# Generate inputs
expiry = rng.uniform(1.0 / 12.0, 5.0, (NUM_SAMPLES, 1))
fwd = rng.uniform(-0.01, 0.04, (NUM_SAMPLES, 1))
ln_vol = rng.uniform(0.05, 0.25, (NUM_SAMPLES, 1))  # Specify log-normal vol
alpha = ln_vol * np.power(np.abs(fwd + SHIFT), 1 - BETA)  # Rough order of magnitude for alpha
nu = rng.uniform(0.20, 0.80, (NUM_SAMPLES, 1))
rho = rng.uniform(-0.40, 0.40, (NUM_SAMPLES, 1))
spread = rng.uniform(-300, 300, (NUM_SAMPLES, 1))  # Specify spreads in bps
strike = fwd + spread / 10000
strike = np.maximum(strike, -SHIFT + 10.0 / 10000.0)
print(f"Generate inputs: {tm.time() - t0:.3f} seconds")

# Generate outputs
t0 = tm.time()
fv = np.ndarray((NUM_SAMPLES, 1))
for i_ in range(NUM_SAMPLES):
    fv[i_, 0] = generator.price(expiry[i_, 0], strike[i_, 0], [fwd[i_, 0], alpha[i_, 0], BETA,
                                                               nu[i_, 0], rho[i_, 0]])

print(f'Generate prices: {tm.time() - t0:.3f} seconds')

t0 = tm.time()
normal_vol = []
for i_ in range(NUM_SAMPLES):
    try:
        normal_vol.append(impliedvol(expiry[i_, 0], fwd[i_, 0], strike[i_, 0], fv[i_, 0],
                                      is_call=False))
    except Exception:
        normal_vol.append(-12345.6789)

print(f'Inversion: {tm.time() - t0:.3f} seconds')


# Create input set
x_set_raw = np.column_stack((expiry, fwd, strike, alpha, nu, rho))

# Filter out bad data and create output set
t0 = tm.time()
MIN_VOL = 0.0001
MAX_VOL = 0.1

x_set = []
y_set = []
for i in range(NUM_SAMPLES):
    nvol = normal_vol[i]
    if math.isnan(nvol) is False and MIN_VOL < nvol < MAX_VOL:
        x_set.append(x_set_raw[i])
        y_set.append(nvol)

print(f'Cleansing: {tm.time() - t0:.3f} seconds')

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
NUM_NEURONS = 16
DROP_OUT = 0.00
keras_model = compose_model(in_dimension, 1, hidden_layers, NUM_NEURONS, DROP_OUT)

# Learning rate scheduler
INIT_LR = 1e-1
FINAL_LR = 1e-4
DECAY = 0.97
STEPS = 100
lr_schedule = FlooredExponentialDecay(INIT_LR, FINAL_LR, DECAY, STEPS)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile
keras_model.compile(loss='mse', optimizer=optimizer)
model = LearningModel(keras_model)

# Display optimizer fields
print("Optimizer settings")
optim_fields = optimizer.get_config()
for field, value in optim_fields.items():
    print(field, ":", value)

# Train the network
EPOCHS = 1000
BATCH_SIZE = 1000
model.train(x_set, y_set, EPOCHS, BATCH_SIZE)
