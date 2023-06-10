""" Generate training data for Stochastic Local Vol models. We implement the direct map here.
    Datasets of parameters (inputs) vs prices/implied vols (outputs) are generated to later train
    a network that learns the so-called 'direct' calculation, i.e. prices from parameter. """
import os
from volsurfacegen import stovolfactory
import settings
from tools.filemanager import check_directory
from tools.timer import Stopwatch


# ################ Runtime configuration ##########################################################
MODEL_TYPE = "SABR"
# MODEL_TYPE = "ShiftedSABR"
# MODEL_TYPE = "McShiftedSABR"
# MODEL_TYPE = "FbSABR"
# MODEL_TYPE = "McShiftedZABR"
# MODEL_TYPE = "McShiftedHeston"
NUM_SAMPLES = 100 * 1000
# The 4 parameters below are only relevant for models whose reference is calculated by MC
NUM_EXPIRIES = 10
SURFACE_SIZE = 50
NUM_MC = 50 * 1000 # 100 * 1000
POINTS_PER_YEAR = 20 # 25

print(">> Set up runtime configuration")
project_folder = os.path.join(settings.WORKFOLDER, "stovol")
print("> Project folder: " + project_folder)
data_folder = os.path.join(project_folder, "samples")
print("> Data folder: " + data_folder)
check_directory(data_folder)
print("> Chosen model: " + MODEL_TYPE)
data_file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")

# ################ Select model ###############################################################
generator = stovolfactory.set_generator(MODEL_TYPE, NUM_EXPIRIES, SURFACE_SIZE, NUM_MC,
                                        POINTS_PER_YEAR)

# ################ Generate dataset ###############################################################
print(">> Generate dataset")

print(f"> Generate {NUM_SAMPLES:,} price samples")
timer_gen = Stopwatch("Generating Samples")
timer_gen.trigger()
data_df = generator.generate_samples(NUM_SAMPLES)
timer_gen.stop()

print("> Convert to normal vol and cleanse data")
timer_conv = Stopwatch("Converting Prices")
timer_conv.trigger()
data_df = generator.to_nvol(data_df, cleanse=True)
num_clean = len(data_df.index)
print(f"> Dataset size after cleansing: {num_clean:,}")
timer_conv.stop()

print("> Output to file: " + data_file)
timer_out = Stopwatch("File Output")
timer_out.trigger()
generator.to_file(data_df, data_file)
timer_out.stop()

# Show timers
timer_gen.print()
timer_conv.print()
timer_out.print()
