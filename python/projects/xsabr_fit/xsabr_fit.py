""" Fit ANN to XSABR models """
import os
# import pandas as pd
# import numpy as np
import tensorflow as tf
from HaganSabrGenerator import HaganSabrGenerator, ShiftedHaganSabrGenerator
import settings
from machinelearning.topology import compose_model
from machinelearning.LearningModel import LearningModel
from machinelearning.FlooredExponentialDecay import FlooredExponentialDecay
import tools.FileManager as FileManager

# ToDo: Pass boolean for put everywhere right after the strike?
# ToDo: Improve call-back and model classes. Base class that does nothing special as call-back,
#       and other base that outputs the LR only
# ToDo: Display result and metrics

# ################ Runtime configuration ##########################################################
data_folder = os.path.join(settings.WORKFOLDER, "XSABRSamples")
print("Looking for data folder " + data_folder)
FileManager.check_directory(data_folder)
# MODEL_TYPE = "HaganSABR"
MODEL_TYPE = "ShiftedHaganSABR"
data_file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")

# Generator factory
if MODEL_TYPE == "HaganSABR":
    generator = HaganSabrGenerator()
elif MODEL_TYPE == "ShiftedHaganSABR":
    generator = ShiftedHaganSabrGenerator()
else:
    raise ValueError("Unknown model: " + MODEL_TYPE)

GENERATE_SAMPLES = True
# Generate dataset
if GENERATE_SAMPLES:
    NUM_SAMPLES = 100 * 1000
    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df = generator.generate_samples(NUM_SAMPLES)
    print("Cleansing data")
    data_df = generator.cleanse(data_df, cleanse=True)
    print("Output to file: " + data_file)
    generator.to_file(data_df, data_file)
    print("Complete!")


# Retrieve dataset
print("Reading data from file: " + data_file)
x_set, y_set, data_df = generator.retrieve_datasets(data_file)
input_dim = x_set.shape[1]
output_dim = y_set.shape[1]
print("Input dimension: " + str(input_dim))
print("Output dimension: " + str(output_dim))
print(data_df.head())

# Initialize the model
hidden_layers = ['softplus', 'softplus', 'softplus']
NUM_NEURONS = 16
DROP_OUT = 0.00
keras_model = compose_model(input_dim, output_dim, hidden_layers, NUM_NEURONS, DROP_OUT)

# Learning rate scheduler
INIT_LR = 1e-1
FINAL_LR = 1e-4
DECAY = 0.97
STEPS = 100
lr_schedule = FlooredExponentialDecay(INIT_LR, FINAL_LR, DECAY, STEPS)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Display optimizer fields
print("Optimizer settings")
optim_fields = optimizer.get_config()
for field, value in optim_fields.items():
    print(field, ":", value)

# Compile
keras_model.compile(loss='mse', optimizer=optimizer)
model = LearningModel(keras_model)

# Train the network
EPOCHS = 1000
BATCH_SIZE = 1000
model.train(x_set, y_set, EPOCHS, BATCH_SIZE)

# print(x_set)
# print(y_set.shape)
# print(y_set)
