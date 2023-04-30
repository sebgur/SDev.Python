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
from machinelearning.callbacks import SDevPyCallback, RefCallback
from machinelearning.datasets import prepare_sets
from tools.filemanager import check_directory
from tools.timer import Stopwatch
from maths.metrics import rmse, tf_rmse
from projects.xsabr import xsabrplot as xplt

# ToDo: Implement custom loss function and check if there's a loss of runtime
# ToDo: Periodical RMSE on test set while training (in callback)
# ToDo: Export trained model to file
# ToDo: Optional reading of model from file
# ToDo: Visual comparison on shifted BS vols for more familiar demo
# ToDo: Finalize and fine-train models on extended parameter range
# ToDo: Put SABR and FB-SABR MC into analytics\fbsabr.py
# ToDo: Import/translate Kienitz's PDEs from C#, especially if we have ZABR?
# ToDo: Store models in git and data in Kaggle

# ################ Runtime configuration ##########################################################
# MODEL_TYPE = "SABR"
MODEL_TYPE = "ShiftedSABR"
GENERATE_SAMPLES = False
NUM_SAMPLES = 100 * 1000
TRAIN_PERCENT = 0.90
TRAIN = True
EPOCHS = 100
BATCH_SIZE = 1000

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
    TRS = TRAIN_PERCENT * 100
    print(f"> Splitting between training set ({TRS:.2f}%) and test set ({100 - TRS:.2f}%)")
    x_train, y_train, x_test, y_test = prepare_sets(x_set, y_set, TRAIN_PERCENT)

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


    # Customized loss (just scaling RMSE by 10,000)
    def tf_bps_rmse(y_true, y_ref):
        """ RMSE in bps in tensorflow """
        return 10000.0 * tf_rmse(y_true, y_ref)

    def bps_rmse(y_true, y_ref):
        """ RMSE in bps """
        return 10000.0 * rmse(y_true, y_ref)

    # Compile
    print("> Compile model")
    keras_model.compile(loss=tf_bps_rmse, optimizer=optimizer)
    # keras_model.compile(loss='mse', optimizer=optimizer)
    model = LearningModel(keras_model)

    # Callbacks
    EPOCH_SAMPLING = 5
    callback = RefCallback(x_test, y_test, bps_rmse, optimizer=optimizer,
                           epoch_sampling=EPOCH_SAMPLING)
    # callback = SDevPyCallback(optimizer=optimizer, epoch_sampling=EPOCH_SAMPLING)

    # Train the network
    print(">> Training ANN model")
    trn_timer = Stopwatch("Training")
    trn_timer.trigger()
    model.train(x_train, y_train, EPOCHS, BATCH_SIZE, callback)
    trn_timer.stop()
    trn_timer.print()

# Analyse results
print(">> Analyse results")

print("> Performance on testing set")
train_pred = model.predict(x_train)
train_rmse = bps_rmse(train_pred, y_train)
print(f"RMSE on training set: {train_rmse:,.2f}")
test_pred = model.predict(x_test)
test_rmse = bps_rmse(test_pred, y_test)
print(f"RMSE on test set: {test_rmse:,.2f}")

# Generate strike spread axis
NUM_TEST = 100
SPREADS = np.linspace(-300, 300, num=NUM_TEST)

plt.figure(figsize=(18, 10))
plt.subplots_adjust(hspace=0.40)

plt.subplot(2, 3, 1)
PARAMS = { 'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25 }
xplt.strike_ladder(0.25, SPREADS, 0.028, PARAMS, generator, model)
plt.subplot(2, 3, 2)
PARAMS = { 'LnVol': 0.25, 'Beta': 0.5, 'Nu': 0.65, 'Rho': -0.15 }
xplt.strike_ladder(0.50, SPREADS, 0.025, PARAMS, generator, model)
plt.subplot(2, 3, 3)
PARAMS = { 'LnVol': 0.25, 'Beta': 0.5, 'Nu': 0.75, 'Rho': -0.10 }
xplt.strike_ladder(0.75, SPREADS, 0.025, PARAMS, generator, model)
plt.subplot(2, 3, 4)
PARAMS = { 'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.65, 'Rho': 0.00 }
xplt.strike_ladder(1.00, SPREADS, 0.025, PARAMS, generator, model)
plt.subplot(2, 3, 5)
PARAMS = { 'LnVol': 0.30, 'Beta': 0.5, 'Nu': 0.50, 'Rho': 0.15 }
xplt.strike_ladder(2.00, SPREADS, 0.025, PARAMS, generator, model)
plt.subplot(2, 3, 6)
PARAMS = { 'LnVol': 0.35, 'Beta': 0.5, 'Nu': 0.25, 'Rho': 0.25 }
xplt.strike_ladder(5.00, SPREADS, 0.025, PARAMS, generator, model)

plt.show()

# Show training history
hist_epochs, hist_losses, hist_lr = callback.convergence()

plt.figure(figsize=(14, 7))
plt.subplots_adjust(hspace=0.40)

plt.subplot(1, 2, 1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale("log")
plt.plot(hist_epochs, hist_losses)
plt.subplot(1, 2, 2)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.plot(hist_epochs, hist_lr)

plt.show()
