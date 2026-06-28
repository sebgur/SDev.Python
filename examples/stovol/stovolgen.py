""" Generate training data for Stochastic Local Vol models. We implement the direct map here.
    Datasets of parameters (inputs) vs prices/implied vols (outputs) are generated to later train
    a network that learns the so-called 'direct' calculation, i.e. prices from parameter. """
import os
from pathlib import Path
from sdevpy.volatility.mlsurfacegen import stovolfactory
from sdevpy.utilities.timer import Stopwatch
from sdevpy import logger
logger.configure()


# ################ Runtime configuration ##########################################################
n_samples = 100_000
model_type = 'SABR'
# model_type = "McSABR"
# model_type = "FbSABR"
# model_type = "McZABR"
# model_type = "McHeston"
shift = 0.03
use_direct = True

# The parameters below are only relevant for models whose reference is calculated by MC
n_expiries = 10
n_strikes = 5
n_mc = 100 * 1000 # 100 * 1000
points_per_year = 25 # 25
seed = 4549 # 123456789, 6789, 9191, 888, 4321, 100, 4444, 72, 1234, 42

print(">> Set up runtime configuration")
project_path = Path(os.environ.get('SDEVPY_DATA', Path.home() / 'sdevpy'))
print(f"> Project folder: {project_path}")
dataset_path = project_path / "datasets" / "stovol" / ("direct" if use_direct else "inverse") / model_type
print("> Chosen model: " + model_type)
dataset_path.mkdir(parents=True, exist_ok=True)
print(f"> Data folder: {dataset_path}")

# Create generator
# generator = SabrGenerator(shift)
# ################ Select model ###################################################################
generator = stovolfactory.set_generator(model_type, shift, n_expiries, n_strikes, n_mc,
                                        points_per_year, seed)

# ################ Select training ranges #########################################################
# SABR
ranges = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.5],
          'Beta': [0.1, 0.9], 'Nu': [0.1, 1.0], 'Rho': [-0.6, 0.6]}
# # FBSABR
# ranges = {'Ttm': [1.0 / 12.0, 5.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.5],
#           'Beta': [0.25, 0.75], 'Nu': [0.1, 1.0], 'Rho': [-0.6, 0.6]}
# # ZABR
# ranges = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.5],
#           'Beta': [0.1, 0.9], 'Nu': [0.10, 1.0], 'Rho': [-0.6, 0.6],
#           'Gamma': [0.1, 0.9]}
# Heston
# ranges = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041], 'LnVol': [0.05, 0.25],
#           'Kappa': [0.25, 4.00], 'Theta': [0.05**2, 0.25**2], 'Xi': [0.10, 0.50],
#           'Rho': [-0.40, 0.40]}

# ################ Generate dataset ###############################################################
print(">> Generate dataset")
print(f"> Generate {n_samples:,} samples")
timer_gen = Stopwatch("Generating Samples")
timer_gen.trigger()
timer_conv = None
if use_direct:
    data_df = generator.generate_samples(n_samples, ranges)
    timer_gen.stop()
    print("> Convert to normal vol and cleanse data")
    timer_conv = Stopwatch("Converting Prices")
    timer_conv.trigger()
    data_df = generator.to_nvol(data_df, cleanse=True)
    num_clean = len(data_df.index)
    print(f"> Dataset size after cleansing: {num_clean:,}")
    timer_conv.stop()
else:
    spreads = [-200, -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100, 200]
    data_df = generator.generate_samples_inverse(n_samples, ranges, spreads)
    timer_gen.stop()


# ################ Output to file #################################################################
timer_out = Stopwatch("File Output")
timer_out.trigger()
file = dataset_path / f"{n_samples}_{seed}.tsv"
print(f"Output to file: {file}")
generator.to_file(data_df, file)
timer_out.stop()
print("Complete!")

# Show timers
timer_gen.print()
if timer_conv is not None:
    timer_conv.print()
timer_out.print()
