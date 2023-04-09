""" Fit ANN to XSABR models. We implement the direct map here. Datasets of parameters (inputs)
    vs prices/implied vols (outputs) are generated (or read from tsv) to train a network that
    learns the so-colled 'direct' calculation, i.e. prices from parameter. """
import os
import numpy as np
import tensorflow as tf
from sabrgenerator import SabrGenerator, ShiftedSabrGenerator
import settings
from machinelearning.topology import compose_model
from machinelearning.LearningModel import LearningModel
from machinelearning.FlooredExponentialDecay import FlooredExponentialDecay
from tools.filemanager import check_directory

# ToDo: Display result and metrics
# ToDo: Improve call-back and model classes. Base class that does nothing special as call-back,
#       and other base that outputs the LR only
# ToDo: Display history of loss and learning rate
# ToDo: Extend parameter range of the fit, especially for beta
# ToDo: Export trained model to file
# ToDo: Optional reading of model from file
# ToDo: Implement split between training and validation datasets
# ToDo: Compare performance on training vs validation sets
# ToDo: Finalize and fine-train models

# ################ Runtime configuration ##########################################################
print(">> Set up runtime configuration")
project_folder = os.path.join(settings.WORKFOLDER, "xsabr_fit")
print("> Project folder " + project_folder)
data_folder = os.path.join(project_folder, "samples")
print("> Data folder " + data_folder)
check_directory(data_folder)
# MODEL_TYPE = "SABR"
MODEL_TYPE = "ShiftedSABR"
print("> Chosen model: " + MODEL_TYPE)
data_file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")

# Generator factory
if MODEL_TYPE == "SABR":
    generator = SabrGenerator()
elif MODEL_TYPE == "ShiftedSABR":
    generator = ShiftedSabrGenerator()
else:
    raise ValueError("Unknown model: " + MODEL_TYPE)

GENERATE_SAMPLES = False
# Generate dataset
if GENERATE_SAMPLES:
    NUM_SAMPLES = 100 * 1000
    print(f">> Generating {NUM_SAMPLES:,} samples")
    data_df = generator.generate_samples(NUM_SAMPLES)
    print("> Convert to normal vol and cleanse data")
    data_df = generator.to_nvol(data_df, cleanse=True)
    print("> Output to file: " + data_file)
    generator.to_file(data_df, data_file)


# Retrieve dataset
print(">> Reading dataset from file: " + data_file)
x_set, y_set, data_df = generator.retrieve_datasets(data_file)
input_dim = x_set.shape[1]
output_dim = y_set.shape[1]
print("> Input dimension: " + str(input_dim))
print("> Output dimension: " + str(output_dim))
print("> Dataset extract")
print(data_df.head())

# Initialize the model
print(">> Compose ANN model")
hidden_layers = ['softplus', 'softplus', 'softplus']
NUM_NEURONS = 16
DROP_OUT = 0.00
keras_model = compose_model(input_dim, output_dim, hidden_layers, NUM_NEURONS, DROP_OUT)
print(f"> Hidden layer structure: {hidden_layers}")
print(f"> Number of neurons per layer: {NUM_NEURONS}")
print(f"> Drop-out rate: {DROP_OUT:.2f}")

# Learning rate scheduler
INIT_LR = 1e-1
FINAL_LR = 1e-4
DECAY = 0.97
STEPS = 100
lr_schedule = FlooredExponentialDecay(INIT_LR, FINAL_LR, DECAY, STEPS)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
print("> Optimizer settings")
optim_fields = optimizer.get_config()
for field, value in optim_fields.items():
    print(field, ":", value)

# Compile
print("> Compile model")
keras_model.compile(loss='mse', optimizer=optimizer)
model = LearningModel(keras_model)

# Train the network
print(">> Training ANN model")
EPOCHS = 50
BATCH_SIZE = 1000
print(f"> Epochs: {EPOCHS:,}")
print(f"> Batch size: {BATCH_SIZE:,}")
model.train(x_set, y_set, EPOCHS, BATCH_SIZE)

# Plot results
print(">> Analyse results")
# Generate strike axis
NUM_TEST = 100
spread_grid = np.linspace(-300, 300, num=NUM_TEST)
# TEST_T = 1.2
# TEST_FWD = 0.035
# TEST_LN_VOL = 0.20
# TEST_BETA = 0.5
# TEST_NU = 0.55
# TEST_RHO = -0.25
TEST_PARAMS = { 'TTM': 1.2, 'F': 0.035, 'LNVOL': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25 }

print(TEST_PARAMS.TTM)

