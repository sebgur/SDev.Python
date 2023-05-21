""" Smile generator for SABR model (shifted or not) using Monte-Carlo to calculate option
    prices """
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
import settings
from analytics import mcsabr
# from analytics import black
# from analytics import bachelier
from volsurfacegen.sabrgenerator import SabrGenerator
from tools import filemanager
from tools import constants
from tools import timer


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

        # Derive number of surfaces to generate
        num_surfaces = int(num_samples / self.surface_size + 1)
        print(f"Number of surfaces to generate: {num_surfaces:,}")

        # Draw parameters
        lnvol = self.rng.uniform(0.05, 0.25, num_surfaces)
        beta = self.rng.uniform(0.49, 0.51, num_surfaces)
        nu = self.rng.uniform(0.20, 0.80, num_surfaces)
        rho = self.rng.uniform(-0.40, 0.40, num_surfaces)

        # Calculate prices per surface
        ts = []
        strikes = []
        fwds = []
        lnvols = []
        betas = []
        nus = []
        rhos = []
        prices = []
        for j in range(num_surfaces):
            print(f"Surface generation number {j+1:,}/{num_surfaces:,}")
            expiries = self.rng.uniform(1.0 / 12.0, 5.0, self.num_expiries)
            fwd = self.rng.uniform(self.min_fwd, self.max_fwd, 1)[0]
            vol = lnvol[j]

            # print("t shape: ", expiries.shape)
            # print("t\n", expiries)
            # print("fwd: ", fwd)

            # Draw strikes
            ks = []
            for expIdx in range(self.num_expiries):
                # Spread method
                # spread = self.rng.uniform(-300, 300, self.num_strikes)
                # k = fwd + spread / 10000.0
                # k = np.maximum(k, -shift + constants.BPS10)

                # Percentile method
                stdev = vol * np.sqrt(expiries[expIdx])
                percentiles = self.rng.uniform(0.01, 0.99, self.num_strikes)
                k = (fwd + shift) * np.exp(-0.5 * stdev * stdev + stdev * sp.norm.ppf(percentiles))
                k = k - shift

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
    NUM_SAMPLES = 100 * 1000
    NUM_MC = 100 * 1000
    POINTS_PER_YEAR = 25
    SURFACE_SIZE = 1000
    NUM_EXPIRIES = 25
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    MODEL_TYPE = 'McShiftedSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "xsabr")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_test.tsv")
    generator = McShiftedSabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

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
