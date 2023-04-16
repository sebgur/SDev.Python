""" Fit ANN to XSABR models. We implement the direct map here. Datasets of parameters (inputs)
    vs prices/implied vols (outputs) are generated (or read from tsv) to train a network that
    learns the so-colled 'direct' calculation, i.e. prices from parameter. """
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from volsurfacegen.sabrgenerator import SabrGenerator, ShiftedSabrGenerator
import settings
from machinelearning.topology import compose_model
from machinelearning.learningmodel import LearningModel
from machinelearning.learningschedules import FlooredExponentialDecay
from tools.filemanager import check_directory
from analytics import bachelier
import xsabrplot as xplt

# ToDo: Display result and metrics
# ToDo: Improve call-back and model classes. Base class that does nothing special as call-back,
#       and other base that outputs the LR only. Finalize machinelearning design.
# ToDo: Display history of loss and learning rate
# ToDo: Export trained model to file
# ToDo: Optional reading of model from file
# ToDo: Implement split between training and validation datasets
# ToDo: Compare performance on training vs validation sets
# ToDo: Finalize and fine-train models on extended parameter range
# ToDo: Put FB-SABR MC into analytics\fbsabr.py
# ToDo: Import/translate Kienitz's PDEs from C#, especially if we have ZABR?

# ################ Runtime configuration ##########################################################
# MODEL_TYPE = "SABR"
MODEL_TYPE = "ShiftedSABR"
GENERATE_SAMPLES = False
NUM_SAMPLES = 100 * 1000
TRAIN = True

print(">> Set up runtime configuration")
project_folder = os.path.join(settings.WORKFOLDER, "xsabr")
print("> Project folder " + project_folder)
data_folder = os.path.join(project_folder, "samples")
print("> Data folder " + data_folder)
check_directory(data_folder)
print("> Chosen model: " + MODEL_TYPE)
data_file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")

# Generator factory
if MODEL_TYPE == "SABR":
    generator = SabrGenerator()
elif MODEL_TYPE == "ShiftedSABR":
    generator = ShiftedSabrGenerator()
else:
    raise ValueError("Unknown model: " + MODEL_TYPE)

# Generate dataset
if GENERATE_SAMPLES:
    print(f">> Generating {NUM_SAMPLES:,} samples")
    data_df = generator.generate_samples(NUM_SAMPLES)
    print("> Convert to normal vol and cleanse data")
    data_df = generator.to_nvol(data_df, cleanse=True)
    print("> Output to file: " + data_file)
    generator.to_file(data_df, data_file)


# If training, retrieve dataset, compose model and train
if TRAIN:
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
    EPOCHS = 100
    BATCH_SIZE = 1000
    print(f"> Epochs: {EPOCHS:,}")
    print(f"> Batch size: {BATCH_SIZE:,}")
    model.train(x_set, y_set, EPOCHS, BATCH_SIZE)

# Analyse results
print(">> Analyse results")
# Generate strike spread axis
NUM_TEST = 100
spread_ladder = np.linspace(-300, 300, num=NUM_TEST)
EXPIRY = 1.2
FWD = 0.028
TEST_PARAMS = { 'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25 }
IS_CALL = generator.is_call

xplt.plot_vol_slice(EXPIRY, spread_ladder, FWD, TEST_PARAMS, generator, model)

# # Calculate prices
# rf_prc, md_prc, strikes, sprds = generator.price_strike_ladder(model, EXPIRY, spread_grid,
#                                                                FWD, TEST_PARAMS)

# # Invert to normal vols
# rf_nvols = []
# md_nvols = []
# for i, strike in enumerate(strikes):
#     rf_nvols.append(bachelier.implied_vol(EXPIRY, strike, IS_CALL, FWD, rf_prc[i]))
#     md_nvols.append(bachelier.implied_vol(EXPIRY, strike, IS_CALL, FWD, md_prc[i]))

# # Plot
# plt.xlabel('Spread')
# plt.ylabel('Volatility')
# plt.plot(spread_grid, rf_nvols, color='blue', label='Reference')
# plt.plot(spread_grid, md_nvols, color='red', label='Model')
# plt.legend(loc='upper right')
# plt.show()
