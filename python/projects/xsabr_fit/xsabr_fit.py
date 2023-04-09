""" Fit ANN to XSABR models """
import os
import tensorflow as tf
from HaganSabrGenerator import SabrGenerator, ShiftedSabrGenerator
import settings
from machinelearning.topology import compose_model
from machinelearning.LearningModel import LearningModel
from machinelearning.FlooredExponentialDecay import FlooredExponentialDecay
import tools.FileManager as FileManager

# ToDo: Display result and metrics
# ToDo: Improve call-back and model classes. Base class that does nothing special as call-back,
#       and other base that outputs the LR only
# ToDo: extend parameter range
# ToDo: Export trained model to file
# ToDo: Optional reading of model from file
# ToDo: Finalize and fine-train models

# ################ Runtime configuration ##########################################################
project_folder = os.path.join(settings.WORKFOLDER, "xsabr_fit")
data_folder = os.path.join(project_folder, "samples")
print("Looking for data folder " + data_folder)
FileManager.check_directory(data_folder)
# MODEL_TYPE = "SABR"
MODEL_TYPE = "ShiftedSABR"
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
    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df = generator.generate_samples(NUM_SAMPLES)
    print("Convert to normal vol and cleanse data")
    data_df = generator.to_nvol(data_df, cleanse=True)
    print("Output to file: " + data_file)
    generator.to_file(data_df, data_file)
    print("Sample generation complete!")


# Retrieve dataset
print("Reading dataset from file: " + data_file)
x_set, y_set, data_df = generator.retrieve_datasets(data_file)
input_dim = x_set.shape[1]
output_dim = y_set.shape[1]
print("Input dimension: " + str(input_dim))
print("Output dimension: " + str(output_dim))
print("Dataset extract")
print(data_df.head())

# Initialize the model
hidden_layers = ['softplus', 'softplus', 'softplus']
NUM_NEURONS = 16
DROP_OUT = 0.00
keras_model = compose_model(input_dim, output_dim, hidden_layers, NUM_NEURONS, DROP_OUT)
# print("Hidden layer structure: " + hidden_layers)
print(f"Number of neurons per layer: {NUM_NEURONS}")
print(f"Drop-out rate: {DROP_OUT:.2f}")

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
EPOCHS = 100
BATCH_SIZE = 1000
model.train(x_set, y_set, EPOCHS, BATCH_SIZE)

# print(x_set)
# print(y_set.shape)
# print(y_set)
