""" Generate training data for Stochastic Local Vol models. We implement the direct map here.
    Datasets of parameters (inputs) vs prices/implied vols (outputs) are generated to later train
    a network that learns the so-called 'direct' calculation, i.e. prices from parameter. """
import os
from volsurfacegen.sabrgenerator import SabrGenerator, ShiftedSabrGenerator
from volsurfacegen.mcsabrgenerator import McShiftedSabrGenerator
from volsurfacegen.fbsabrgenerator import FbSabrGenerator
from volsurfacegen.mczabrgenerator import McShiftedZabrGenerator
from volsurfacegen.mchestongenerator import McShiftedHestonGenerator
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
NUM_SAMPLES = 100 * 1000 # Relevant if GENERATE_SAMPLES is True

print(">> Set up runtime configuration")
project_folder = os.path.join(settings.WORKFOLDER, "xsabr")
print("> Project folder: " + project_folder)
data_folder = os.path.join(project_folder, "samples")
print("> Data folder: " + data_folder)
check_directory(data_folder)
print("> Chosen model: " + MODEL_TYPE)
data_file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")

# ################ Select model ###############################################################
if MODEL_TYPE == "SABR":
    generator = SabrGenerator()
elif MODEL_TYPE == "ShiftedSABR":
    generator = ShiftedSabrGenerator()
elif MODEL_TYPE == "McShiftedSABR":
    NUM_EXPIRIES = 10
    SURFACE_SIZE = 50
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    NUM_MC = 50 * 1000 # 100 * 1000
    POINTS_PER_YEAR = 20 # 25
    generator = McShiftedSabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)
elif MODEL_TYPE == "FbSABR":
    NUM_EXPIRIES = 10
    SURFACE_SIZE = 50
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    NUM_MC = 50 * 1000 # 100 * 1000
    POINTS_PER_YEAR = 20 # 25
    generator = FbSabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)
elif MODEL_TYPE == "McShiftedZABR":
    NUM_EXPIRIES = 10
    SURFACE_SIZE = 50
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    NUM_MC = 50 * 1000 # 100 * 1000
    POINTS_PER_YEAR = 20 # 25
    generator = McShiftedZabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)
elif MODEL_TYPE == "McShiftedHeston":
    NUM_EXPIRIES = 10
    SURFACE_SIZE = 50
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    NUM_MC = 50 * 1000 # 100 * 1000
    POINTS_PER_YEAR = 20 # 25
    generator = McShiftedHestonGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)
else:
    raise ValueError("Unknown model: " + MODEL_TYPE)

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
