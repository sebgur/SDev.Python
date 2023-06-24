""" Smile generator for Free-Boundary SABR (FBSABR) using Monte-Carlo for option prices """
import os
from sdevpy import settings
from sdevpy.analytics import fbsabr
from sdevpy.volsurfacegen.mcsabrgenerator import McSabrGenerator
from sdevpy.tools import filemanager
from sdevpy.tools import timer


class FbSabrGenerator(McSabrGenerator):
    """ Free-Boundary SABR model using Monte-Carlo to calculate option prices. """
    def __init__(self, num_expiries=15, num_strikes=10, num_mc=10000, points_per_year=10,
                 seed=42):
        McSabrGenerator.__init__(self, 0.03, num_expiries, num_strikes, num_mc,
                                 points_per_year, seed)

    def price(self, expiries, strikes, are_calls, fwd, parameters):
        prices = fbsabr.price(expiries, strikes, are_calls, fwd, parameters,
                              self.num_mc, self.points_per_year)

        return prices


if __name__ == "__main__":
    NUM_SAMPLES = 1 * 1000
    NUM_MC = 1 * 1000
    POINTS_PER_YEAR = 25
    SURFACE_SIZE = 50
    NUM_EXPIRIES = 10
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    MODEL_TYPE = 'FbSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "stovol")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_tests.tsv")
    generator = FbSabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

    ranges = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041],
              'LnVol': [0.05, 0.5], 'Beta': [0.1, 0.9], 'Nu': [0.1, 1.0], 'Rho': [-0.6, 0.6]}
    print("Generating " + str(NUM_SAMPLES) + " samples")
    gen_timer = timer.Stopwatch("Sample Generation")
    gen_timer.trigger()
    data_df_ = generator.generate_samples(NUM_SAMPLES, ranges)
    gen_timer.stop()
    print(data_df_)
    print("Cleansing data")
    nvol_timer = timer.Stopwatch("Convertion to normal vols")
    nvol_timer.trigger()
    data_df_ = generator.to_nvol(data_df_)
    nvol_timer.stop()
    print("Output to file: " + file)
    file_timer = timer.Stopwatch("Output to file")
    file_timer.trigger()
    generator.to_file(data_df_, file)
    file_timer.stop()
    print("Complete!")
    gen_timer.print()
    nvol_timer.print()
    file_timer.print()
