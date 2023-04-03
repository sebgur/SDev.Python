import settings
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from HaganSabrGenerator import HaganSabrGenerator, ShiftedHaganSabrGenerator
from machinelearning.topology import compose_model
from machinelearning.LearningModel import LearningModel
from machinelearning.FlooredExponentialDecay import FlooredExponentialDecay

# ################ ToDo ###################################################################################
# Refresh marchine learning py
# Implement generic training
# Display result and metrics

# ################ Runtime configuration ##################################################################
data_folder = os.path.join(settings.workfolder, "XSABRSamples")
model_type = "ShiftedHaganSABR"

# Generator factory
if model_type == "HaganSABR":
    generator = HaganSabrGenerator()
elif model_type == "ShiftedHaganSABR":
    generator = ShiftedHaganSabrGenerator()
else:
    raise Exception("Unknown model: " + model_type)


# Retrieve dataset
data_file = os.path.join(data_folder, model_type + "_samples.tsv")
print("Reading data from file: " + data_file)
x_set, y_set, data_df = generator.retrieve_datasets(data_file)
input_dim = x_set.shape[1]
output_dim = y_set.shape[1]
print("Input dimension: " + str(input_dim))
print("Output dimension: " + str(output_dim))
print(data_df.head())

# Initialize the model
hidden_layers = ['softplus', 'softplus', 'softplus']
num_neurons = 16
dropout = 0.00
keras_model = compose_model(input_dim, output_dim, hidden_layers, num_neurons, dropout)

# Learning rate scheduler
init_lr = 1e-1
final_lr = 1e-4
decay = 0.97
steps = 100
lr_schedule = FlooredExponentialDecay(init_lr, final_lr, decay, steps)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# Display optimizer fields
print("Optimizer settings")
optim_fields = optimizer.get_config()
for field in optim_fields:
    print(field, ":", optim_fields[field])

# Compile
keras_model.compile(loss='mse', optimizer=optimizer)
model = LearningModel(keras_model)

# Train the network
epochs = 1000
batch_size = 1000
history = model.train(x_set, y_set, epochs, batch_size)

# print(x_set)
# print(y_set.shape)
# print(y_set)

