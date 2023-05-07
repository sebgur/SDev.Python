""" Smile generator for SABR model (shifted or not) using Monte-Carlo to calculate option
    prices """
import os
import numpy as np
import pandas as pd
import settings
from analytics import mcsabr
# from analytics import black
# from analytics import bachelier
from volsurfacegen.sabrgenerator import SabrGenerator
from tools import filemanager
from tools import constants


# ToDo: Run a couple more MC runs with full accuracy and check on s/s against Hagan
# ToDo: Plug cleansing/inversion to nvol and check on s/s
# ToDo: Try full training
# ToDo: Test if better to use smaller numbers of expiries to have more diversity in
#       sampled fwds and parameters.

class McSabrGenerator(SabrGenerator):
    """ SABR model with a generic shift value, using Monte-Carlo to calculate option prices. """
    def __init__(self, shift=0.0, num_expiries=15, num_strikes=10, num_mc=10000,
                 points_per_year=10):
        SabrGenerator.__init__(self, shift)
        self.num_strikes = num_strikes
        self.num_expiries = num_expiries
        self.surface_size = self.num_expiries * self.num_strikes
        self.num_mc = num_mc
        self.points_per_year = points_per_year
        self.are_calls = [[self.is_call] * self.num_strikes] * self.num_expiries

    def generate_samples(self, num_samples):
        shift = self.shift

        print(f"Number of strikes: {self.num_strikes:,}")
        print(f"Number of expiries: {self.num_expiries:,}")
        print(f"Surface size: {self.surface_size:,}")
        print(f"Number of samples: {num_samples:,}")

        # Derive number of random expiries
        num_mc_runs = int(num_samples / self.surface_size) # ToDo: + 1
        print(f"Number of MC runs: {num_mc_runs:,}")

        # Draw parameters
        lnvol = self.rng.uniform(0.05, 0.25, num_mc_runs)
        beta = self.rng.uniform(0.49, 0.51, num_mc_runs)
        nu = self.rng.uniform(0.20, 0.80, num_mc_runs)
        rho = self.rng.uniform(-0.40, 0.40, num_mc_runs)

        # Calculate prices
        ts = []
        strikes = []
        fwds = []
        lnvols = []
        betas = []
        nus = []
        rhos = []
        prices = []
        for j in range(num_mc_runs):
            print(f"MC simulation {j+1:,}/{num_mc_runs:,}")
            expiries = self.rng.uniform(1.0 / 12.0, 5.0, self.num_expiries)
            fwd = self.rng.uniform(self.min_fwd, self.max_fwd, 1)[0]

            # print("t shape: ", expiries.shape)
            # print("t\n", expiries)
            # print("fwd: ", fwd)

            # Draw strikes
            ks = []
            for _ in range(self.num_expiries):
                spread = self.rng.uniform(-300, 300, self.num_strikes)
                k = fwd + spread / 10000.0
                k = np.maximum(k, -shift + constants.BPS10)
                ks.append(k)

            ks = np.asarray(ks)
            # print("Strikes shape: ", ks.shape)
            # print("Strikes\n", ks)

            # Draw parameters
            params = {'LnVol': lnvol[j], 'Beta': beta[j], 'Nu': nu[j], 'Rho': rho[j]}

            # Calculate prices
            price = self.price(expiries, ks, self.are_calls, fwd, params)
            # print(price)

            # Flatten the results
            for exp_idx, expiry in enumerate(expiries):
                ts.extend([expiry] * self.num_strikes)
                strikes.extend(ks[exp_idx])
                fwds.extend([fwd] * self.num_strikes)
                lnvols.extend([lnvol[j]] * self.num_strikes)
                betas.extend([beta[j]] * self.num_strikes)
                nus.extend([nu[j]] * self.num_strikes)
                rhos.extend([rho[j]] * self.num_strikes)
                prices.extend(price[exp_idx])


            # print("expiries\n", ts)
            # print("strikes\n", strikes)
            # print("fwds\n", fwds)

        # Put in dataframe
        df = pd.DataFrame({'Ttm': ts, 'K': strikes, 'F': fwds, 'LnVol': lnvols, 'Beta': betas,
                           'Nu': nus, 'Rho': rhos, 'Price': prices})
        df.columns = ['Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Price']

        return df

    def price(self, expiry, strike, is_call, fwd, parameters):
        shifted_k = strike + self.shift
        shifted_f = fwd + self.shift
        prices = mcsabr.price(expiry, shifted_k, is_call, shifted_f, parameters,
                              self.num_mc, self.points_per_year)

        return prices


# Special class for Shifted SABR with shift = 3%, for easier calling
class McShiftedSabrGenerator(McSabrGenerator):
    """ For calling convenience, derived from McSabrGenerator with shift at typical 3%. """
    def __init__(self, num_expiries=15, num_strikes=10, num_mc=10000, points_per_year=10):
        McSabrGenerator.__init__(self, 0.03, num_expiries, num_strikes, num_mc, points_per_year)


if __name__ == "__main__":
    NUM_SAMPLES = 20 #100 * 1000
    NUM_MC = 1000
    POINTS_PER_YEAR = 2
    NUM_EXPIRIES = 3
    NUM_STRIKES = 5
    MODEL_TYPE = 'McShiftedSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "xsabr")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_test.tsv")
    generator = McShiftedSabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df_ = generator.generate_samples(NUM_SAMPLES)
    print(data_df_)
    # print("Cleansing data")
    # data_df_ = generator.to_nvol(data_df_)
    print("Output to file: " + file)
    generator.to_file(data_df_, file)
    print("Complete!")
