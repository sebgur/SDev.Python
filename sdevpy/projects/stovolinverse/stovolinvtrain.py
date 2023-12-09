""" Train ANN on datasets for Stochastic Local Vol models. We implement the inverse map here.
    Datasets of prices (inputs) vs parameters (outputs) have been generated
    in a previous set and are now read from tsv. The network here is either loaded from a
    pre-trained state or trained from scratch. Pre-trained models can be loaded and training
    resumed. """
import os
from sdevpy import settings
from datetime import datetime
import numpy as np
# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import tensorflow as tf
# tf.get_logger().setLevel(tf._logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
import matplotlib.pyplot as plt
from sdevpy.machinelearning.topology import compose_model
from sdevpy.machinelearning.learningmodel import LearningModel, load_learning_model
from sdevpy.machinelearning.learningschedules import FlooredExponentialDecay
from sdevpy.machinelearning.callbacks import RefCallback
from sdevpy.machinelearning import datasets
from sdevpy.tools import filemanager
from sdevpy.tools.timer import Stopwatch
# from sdevpy.tools import clipboard
from sdevpy.maths.metrics import bps_rmse, tf_bps_rmse
from sdevpy.volsurfacegen.stovolfactory import set_generator
from sdevpy.projects.stovol import stovolplot as xplt


# ################ ToDo ###########################################################################
# Retrieve data: need to find a way to split between inputs an outputs. That should easy to do
# by counting the headers, but that won't tell us the values of the spread. For now, we'll have
# to assume them by hard-coding. But we could store this information in the additional json
# like we were using in the direct map?
# Train network on prices as inputs, SABR as output
# Compare against solving with optimizer
# This method should allow us to calculate the sensitivity to market by AAD. That's quite an 
# advantage, as for the regular method we can't AAD through calibration/optimization.
# Compare vegas

# ################ Module versions ################################################################
print("TensorFlow version: " + tf.__version__)
# print("Keras version: " + tf.keras.__version__)
print("NumPy version: " + np.__version__)
# print("SciPy version: " + scipy.__version__)
# print("SciKit version: " + sk.__version__)

# ################ Runtime configuration ##########################################################
MODEL_TYPE = "SABR"
# MODEL_TYPE = "McSABR"
# MODEL_TYPE = "FbSABR"
# MODEL_TYPE = "McZABR"
# MODEL_TYPE = "McHeston"
# MODEL_ID = "SABR_3L_64n" # For pre-trained model ID (we can pre-train several versions)
MODEL_ID = MODEL_TYPE # For pre-trained model ID (we can pre-train several versions)
SHIFT = 0.03
USE_TRAINED = False
DOWNLOAD_MODELS = False # Only used when USE_TRAINED is True
DOWNLOAD_DATASETS = False # Use when already created/downloaded
TRAIN = True
if USE_TRAINED is False and TRAIN is False:
    raise RuntimeError("When not using pre-trained models, a new model must be trained")

NUM_SAMPLES = 500 * 1000 # Number of samples to read from sample files
TRAIN_PERCENT = 0.90 # Proportion of dataset used for training (rest used for test)
EPOCHS = 2000
BATCH_SIZE = 1000
SHOW_VOL_CHARTS = True # Show smile section charts
# For comparison to reference values (accuracy of reference)
NUM_MC = 100 * 1000 # 100 * 1000
POINTS_PER_YEAR = 25# 25
project_folder = os.path.join(settings.WORKFOLDER, "stovolinv")

print(">> Set up runtime configuration")
print("> Chosen model type: " + MODEL_TYPE)
if USE_TRAINED:
    print("> Pre-trained model ID: " + MODEL_ID)

filemanager.check_directory(project_folder)
print("> Project folder: " + project_folder)

dataset_folder = os.path.join(project_folder, "datasets")
data_folder = os.path.join(dataset_folder, MODEL_TYPE)
filemanager.check_directory(data_folder)
print("> Data folder: " + data_folder)
data_file = os.path.join(dataset_folder, MODEL_TYPE + "_data.tsv")
print("> Data file: " + data_file)

model_folder = os.path.join(project_folder, "models")
filemanager.check_directory(model_folder)
print("> Model folder: " + model_folder)

if USE_TRAINED and DOWNLOAD_MODELS:
    url = 'https://github.com/sebgur/SDev.Python/raw/main/models/stovolinv/models.zip'
    print("> Downloading and unzipping models from: " + url)
    filemanager.download_unzip(url, model_folder)

if DOWNLOAD_DATASETS:
    url = 'https://github.com/sebgur/SDev.Python/raw/main/datasets/stovolinv/datasets.zip'
    print("> Downloading and unzipping datasets from: " + url)
    filemanager.download_unzip(url, dataset_folder)

# ################ Select generator ###############################################################
# Select generator. The number of expiries and surface size are irrelevant as here we do not
# generate sample data but read it from files. Number of MC and points per year are required
# to calculate the reference values against which we can validate the model.
generator = set_generator(MODEL_TYPE, shift=SHIFT, num_mc=NUM_MC, points_per_year=POINTS_PER_YEAR)

# ################ Prepare datasets ###############################################################
# Datasets are always read, as even if we don't train, we're still going to evaluate the
# performance of the pre-trained model
print(">> Preparing datasets")
# Retrieve data from dataset folder
print(f"> Requested {NUM_SAMPLES:,} samples")
datasets.retrieve_data(data_folder, NUM_SAMPLES, shuffle=True, export_file=data_file)
print("> Exporting dataset to file: " + data_file)
# Retrieve dataset
print("> Reading dataset from file: " + data_file)
x_set, y_set, data_df = generator.retrieve_inverse_datasets(data_file, shuffle=True)
input_dim = x_set.shape[1]
output_dim = y_set.shape[1]
print("> Input dimension: " + str(input_dim))
print("> Output dimension (parameters): " + str(output_dim))
print("> Dataset extract")
print(data_df.head())
# Split into training and test sets
TRS = TRAIN_PERCENT * 100
print(f"> Splitting between training set ({TRS:.2f}%) and test set ({100 - TRS:.2f}%)")
x_train, y_train, x_test, y_test = datasets.prepare_sets(x_set, y_set, TRAIN_PERCENT)
print(f"> Training set size: {x_train.shape[0]:,}")
print(f"> Testing set size: {x_test.shape[0]:,}")

# ################ Compose/Load the model #########################################################
# Compose new model or load pre-trained one
if USE_TRAINED:
    print(">> Loading pre-trained model")
    model_folder_name = os.path.join(model_folder, MODEL_ID)
    print("> Loading pre-trained model from: " + model_folder_name)
    model = load_learning_model(model_folder_name)
    keras_model = model.model
    HIDDEN_LAYERS = NUM_NEURONS = DROP_OUT = None
    topology = model.topology_
    if topology is not None:
        HIDDEN_LAYERS = topology['layers']
        NUM_NEURONS = topology['neurons']
        DROP_OUT = topology['dropout']
else:
    print(">> Composing new model")
    # Initialize the model
    HIDDEN_LAYERS = ['softplus', 'softplus', 'softplus']
    NUM_NEURONS = 64
    DROP_OUT = 0.0
    keras_model = compose_model(input_dim, output_dim, HIDDEN_LAYERS, NUM_NEURONS, DROP_OUT)
    topology = { 'layers': HIDDEN_LAYERS, 'neurons': NUM_NEURONS, 'dropout': DROP_OUT}

    model = LearningModel(keras_model)
    model.topology_ = topology

# Display topology
print(f"> Hidden layer structure: {HIDDEN_LAYERS}")
print(f"> Number of neurons per layer: {NUM_NEURONS}")
print(f"> Drop-out rate: {DROP_OUT:.2f}")

# ################ Train the model ################################################################
if TRAIN:
    # Learning rate scheduler
    INIT_LR = 1.0e-2
    FINAL_LR = 1.0e-4
    DECAY = 0.97
    STEPS = 250
    lr_schedule = FlooredExponentialDecay(INIT_LR, FINAL_LR, DECAY, STEPS)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.optimizer_ = optimizer.get_config()
    print("> Optimizer settings")
    optim_fields = model.optimizer_
    for field, value in optim_fields.items():
        print("> ", field, ":", value)

    # Compile
    print("> Compile model")
    keras_model.compile(loss=tf_bps_rmse, optimizer=optimizer)

    # Callbacks
    EPOCH_SAMPLING = 5
    callback = RefCallback(x_test, y_test, bps_rmse, optimizer=optimizer,
                           epoch_sampling=EPOCH_SAMPLING)

    # Train the network
    print(">> Training ANN model")
    trn_timer = Stopwatch("Training")
    trn_timer.trigger()
    model.train(x_train, y_train, EPOCHS, BATCH_SIZE, callback)
    trn_timer.stop()
    trn_timer.print()

    # Save trained model to file
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d-%H_%M_%S")
    model_folder_name = os.path.join(model_folder, MODEL_TYPE + "_" + dt_string)
    print("Saving model to: " + model_folder_name)
    model.save(model_folder_name)


# ################ Performance analysis ###########################################################
# Analyse results
print(">> Analyse results")

# Check performance
train_pred = model.predict(x_train)
train_rmse = bps_rmse(train_pred, y_train)
print(f"> RMSE(nvol) on training set: {train_rmse:,.2f}")

test_pred = model.predict(x_test)
test_rmse = bps_rmse(test_pred, y_test)
print(f"> RMSE(nvol) on test set: {test_rmse:,.2f}")

# Generate strike spread axis
if SHOW_VOL_CHARTS:
    print("> Choosing a sample parameter set to display chart")
    NUM_STRIKES = 100
    PARAMS = { 'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25, 'Gamma': 0.7, 'Kappa': 1.0,
                'Theta': 0.03, 'Xi': 0.35 }
    FWD = 0.028

    # Any number of expiries can be calculated, but for optimum display choose no more than 6
    if MODEL_TYPE == "FbSABR":
        EXPIRIES = np.asarray([0.25, 0.50, 1.0, 2.0, 5.0, 10.0]).reshape(-1, 1) # Only trained up to 5y
    else:
        EXPIRIES = np.asarray([0.25, 0.50, 1.0, 5.0, 10.0, 30.0]).reshape(-1, 1)
    NUM_EXPIRIES = EXPIRIES.shape[0]
    METHOD = 'Percentiles'
    PERCENTS = np.linspace(0.01, 0.99, num=NUM_STRIKES)
    PERCENTS = np.asarray([PERCENTS] * NUM_EXPIRIES)

    strikes = generator.convert_strikes(EXPIRIES, PERCENTS, FWD, PARAMS, METHOD)
    # clipboard.export2d(strikes)
    ARE_CALLS = [[False] * NUM_STRIKES] * NUM_EXPIRIES # All puts
    # ARE_CALLS = [[False if s < FWD else True for s in expks] for expks in strikes] # Puts/calls
    # print(ARE_CALLS)

    print("> Calculating chart surface with reference model")
    timer_ref = Stopwatch("Reference surface calculation")
    timer_ref.trigger()
    ref_prices = generator.price_surface_ref(EXPIRIES, strikes, ARE_CALLS, FWD, PARAMS)
    timer_ref.stop()
    # clipboard.export2d(ref_prices)
    print("> Calculating chart surface with trained model")
    timer_mod = Stopwatch("Model surface calculation")
    timer_mod.trigger()
    mod_prices = generator.price_surface_mod(model, EXPIRIES, strikes, ARE_CALLS, FWD, PARAMS)
    timer_mod.stop()
    # clipboard.export2d(mod_prices)
    print(f"> Ref-Mod RMSE(price): {bps_rmse(ref_prices, mod_prices):.2f}")

    # Display timers
    timer_ref.print()
    timer_mod.print()

    # Available tranforms: Price, ShiftedBlackScholes, Bachelier
    TITLE = f"{MODEL_TYPE} smile sections, forward={FWD*100:.2f}"#,%\n parameters={PARAMS}"
    # TRANSFORM = "Bachelier"
    # TRANSFORM = "Price"
    TRANSFORM = "ShiftedBlackScholes"
    xplt.plot_transform_surface(EXPIRIES, strikes, ARE_CALLS, FWD, ref_prices, mod_prices,
                                TITLE, transform=TRANSFORM)

# Show training history
if TRAIN:
    hist_epochs = callback.epochs
    hist_losses = callback.losses
    hist_lr = callback.learning_rates
    sampled_epochs = callback.sampled_epochs
    test_losses = callback.test_losses

    plt.figure(figsize=(14, 7))
    plt.subplots_adjust(hspace=0.40)

    plt.subplot(1, 2, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.plot(hist_epochs, hist_losses, label='Loss on training set')
    plt.plot(sampled_epochs, test_losses, color='red', label='Loss on test set')
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.plot(hist_epochs, hist_lr)

    plt.show()
