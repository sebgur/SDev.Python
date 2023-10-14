""" Generate training data for Stochastic Local Vol models. We implement the inverse map here.
    Datasets of parameters (outputs) vs prices/implied vols (inputs) are generated to later train
    a network that learns the so-called 'inverse' calculation, i.e. parameters from prices. """
import os
from datetime import datetime
from sdevpy.volsurfacegen import stovolfactory
from sdevpy import settings
from sdevpy.tools import filemanager
from sdevpy.tools.timer import Stopwatch

# ################ ToDo ###########################################################################
# Generate random sets of SABR parameters
# Calculate output prices on fixed spread grid (possibly with holes?)
# Train network on prices as inputs, SABR as output
# Compare against solving with optimizer
# Compare vegas

# ################ Runtime configuration ##########################################################
MODEL_TYPE = "SABR"
# MODEL_TYPE = "McSABR"
# MODEL_TYPE = "FbSABR"
# MODEL_TYPE = "McZABR"
# MODEL_TYPE = "McHeston"
SHIFT = 0.03
NUM_SAMPLES = 1 * 100
# The 4 parameters below are only relevant for models whose reference is calculated by MC
NUM_EXPIRIES = 10
NUM_STRIKES = 5
NUM_MC = 100 * 1000 # 100 * 1000
POINTS_PER_YEAR = 25 # 25
SEED = 4549# [123456789, 6789, 9191, 888, 4321, 100, 4444, 72, 1234, 42]

print(">> Set up runtime configuration")
project_folder = os.path.join(settings.WORKFOLDER, "stovol")
print("> Project folder: " + project_folder)
dataset_folder = os.path.join(project_folder, "datasets")
print("> Data folder: " + dataset_folder)
filemanager.check_directory(dataset_folder)
print("> Chosen model: " + MODEL_TYPE)
# data_file = os.path.join(dataset_folder, MODEL_TYPE + "_data.tsv")

# ################ Select model ###################################################################
generator = stovolfactory.set_generator(MODEL_TYPE, SHIFT, NUM_EXPIRIES, NUM_STRIKES, NUM_MC,
                                        POINTS_PER_YEAR, SEED)

# ################ Select training ranges #########################################################
# SABR
RANGES = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.5],
          'Beta': [0.1, 0.9], 'Nu': [0.1, 1.0], 'Rho': [-0.6, 0.6]}
# # FBSABR
# RANGES = {'Ttm': [1.0 / 12.0, 5.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.5],
#           'Beta': [0.25, 0.75], 'Nu': [0.1, 1.0], 'Rho': [-0.6, 0.6]}
# # ZABR
# RANGES = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.5],
#           'Beta': [0.1, 0.9], 'Nu': [0.10, 1.0], 'Rho': [-0.6, 0.6],
#           'Gamma': [0.1, 0.9]}
# Heston
# RANGES = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.25],
#           'Kappa': [0.25, 4.00], 'Theta': [0.05**2, 0.25**2], 'Xi': [0.10, 0.50],
#           'Rho': [-0.40, 0.40]}

# ################ Generate dataset ###############################################################
print(">> Generate dataset")

print(f"> Generate {NUM_SAMPLES:,} price samples")
timer_gen = Stopwatch("Generating Samples")
timer_gen.trigger()
data_df = generator.generate_samples(NUM_SAMPLES, RANGES)
# full_data_file = os.path.join(data_folder, MODEL_TYPE + "_samples_full.tsv")
# generator.to_file(data_df, full_data_file)
timer_gen.stop()

print("> Convert to normal vol and cleanse data")
timer_conv = Stopwatch("Converting Prices")
timer_conv.trigger()
data_df = generator.to_nvol(data_df, cleanse=True)
num_clean = len(data_df.index)
print(f"> Dataset size after cleansing: {num_clean:,}")
timer_conv.stop()

timer_out = Stopwatch("File Output")
timer_out.trigger()
now = datetime.now()
dt_string = now.strftime("%Y%m%d-%H_%M_%S")
data_file = os.path.join(dataset_folder, MODEL_TYPE + "_data_" + dt_string + ".tsv")
print("> Output to file: " + data_file)
generator.to_file(data_df, data_file)
timer_out.stop()

# Show timers
timer_gen.print()
timer_conv.print()
timer_out.print()
