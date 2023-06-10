""" Smile generator for Free-Boundary SABR model (FBSABR) using Monte-Carlo to calculate option
    prices """
import os
import settings
from analytics import fbsabr
from volsurfacegen.mcsabrgenerator import McSabrGenerator
from tools import filemanager
from tools import timer


class FbSabrGenerator(McSabrGenerator):
    """ Free-Boundary SABR model using Monte-Carlo to calculate option prices. """
    def __init__(self, num_expiries=15, num_strikes=10, num_mc=10000, points_per_year=10):
        McSabrGenerator.__init__(self, 0.03)
        self.num_strikes = num_strikes
        self.num_expiries = num_expiries
        self.surface_size = self.num_expiries * self.num_strikes
        self.num_mc = num_mc
        self.points_per_year = points_per_year
        self.are_calls = [[self.is_call] * self.num_strikes] * self.num_expiries

    def price(self, expiries, strikes, is_call, fwd, parameters):
        prices = fbsabr.price(expiries, strikes, is_call, fwd, parameters,
                              self.num_mc, self.points_per_year)

        return prices


if __name__ == "__main__":
    NUM_SAMPLES = 100 * 1000
    NUM_MC = 100 * 1000
    POINTS_PER_YEAR = 25
    SURFACE_SIZE = 50
    NUM_EXPIRIES = 10
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    MODEL_TYPE = 'FbSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "stovol")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")
    generator = FbSabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

    print("Generating " + str(NUM_SAMPLES) + " samples")
    gen_timer = timer.Stopwatch("Sample Generation")
    gen_timer.trigger()
    data_df_ = generator.generate_samples(NUM_SAMPLES)
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
