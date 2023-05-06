""" Smile generator for SABR model (shifted or not) using Monte-Carlo to calculate option
    prices """
import os
import numpy as np
import pandas as pd
import settings
# from analytics import sabr
# from analytics import black
# from analytics import bachelier
from volsurfacegen.sabrgenerator import SabrGenerator
# from tools import filemanager
from tools import constants


class McSabrGenerator(SabrGenerator):
    """ SABR model with a generic shift value, using Monte-Carlo to calculate option prices. """
    def __init__(self, shift=0.0, num_expiries=15, num_strikes=10):
        SabrGenerator.__init__(self, shift)
        self.num_strikes = num_strikes
        self.num_expiries = num_expiries
        self.surface_size = self.num_expiries * self.num_strikes

    def generate_samples(self, num_samples):
        shift = self.shift

        # Derive number of random expiries
        num_mc_runs = int(num_samples / self.surface_size)
        print(f"Number of MC runs: {num_mc_runs:,}")

        # Draw parameters
        lnvol = self.rng.uniform(0.05, 0.25, num_mc_runs)
        beta = self.rng.uniform(0.49, 0.51, num_mc_runs)
        nu = self.rng.uniform(0.20, 0.80, num_mc_runs)
        rho = self.rng.uniform(-0.40, 0.40, num_mc_runs)

        for j in range(num_mc_runs):
            t = self.rng.uniform(1.0 / 12.0, 5.0, self.num_expiries)
            fwd = self.rng.uniform(self.min_fwd, self.max_fwd, self.num_expiries)

            # Draw strikes
            strikes = []
            for expiry in range(self.num_expiries):
                spread = self.rng.uniform(-300, 300, self.num_strikes)
                strike = fwd + spread / 10000.0
                strike = np.maximum(strike, -shift + constants.BPS10)
                strikes.append(strike)

            # Draw parameters
            params = {'LnVol': lnvol[j], 'Beta': beta[j], 'Nu': nu[j], 'Rho': rho[j]}

            # Calculate prices
            price = self.price(t, strike, self.is_call, fwd, params)

        # Put in dataframe
        df = pd.DataFrame({'Ttm': t, 'K': strike, 'F': fwd, 'LnVol': lnvol, 'Beta': beta, 'Nu': nu,
                           'Rho': rho, 'Price': price})
        df.columns = ['Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Price']

        return df

    def price(self, expiry, strike, is_call, fwd, parameters):
        shifted_k = strike + self.shift
        shifted_f = fwd + self.shift

        # return price


# Special class for Shifted SABR with shift = 3%, for easier calling
class McShiftedSabrGenerator(McSabrGenerator):
    """ For calling convenience, derived from SabrGenerator with shift at typical 3%. """
    def __init__(self, num_expiries=15, num_strikes=10):
        McSabrGenerator.__init__(self, 0.03, num_expiries, num_strikes)


if __name__ == "__main__":
    NUM_SAMPLES = 10 #100 * 1000
    MODEL_TYPE = 'McShiftedSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "xsabr")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_test.tsv")
    generator = McShiftedSabrGenerator()

    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df_ = generator.generate_samples(NUM_SAMPLES)
    print(data_df_)
    print("Cleansing data")
    data_df_ = generator.to_nvol(data_df_)
    print("Output to file: " + file)
    generator.to_file(data_df_, file)
    print("Complete!")
