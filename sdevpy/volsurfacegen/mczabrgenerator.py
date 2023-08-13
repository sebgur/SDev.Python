""" Smile generator for ZABR model using Monte-Carlo to calculate option prices """
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
from sdevpy import settings
from sdevpy.analytics import mczabr
from sdevpy.analytics import bachelier
from sdevpy.volsurfacegen.smilegenerator import SmileGenerator
from sdevpy.tools import filemanager
from sdevpy.tools import timer


class McZabrGenerator(SmileGenerator):
    """ ZABR model with a generic shift value, using Monte-Carlo to calculate option prices. """
    def __init__(self, shift=0.0, num_expiries=15, num_strikes=10, num_mc=10000,
                 points_per_year=10, seed=42):
        SmileGenerator.__init__(self, shift, num_expiries, num_strikes, seed)
        self.num_mc = num_mc
        self.points_per_year = points_per_year

    def generate_samples(self, num_samples, rg):
        shift = self.shift

        print(f"Number of strikes: {self.num_strikes:,}")
        print(f"Number of expiries: {self.num_expiries:,}")
        print(f"Surface size: {self.surface_size:,}")
        print(f"Number of samples: {num_samples:,}")

        # Derive number of surfaces to generate
        num_surfaces = int(num_samples / self.surface_size)# + 1)
        print(f"Number of surfaces/parameter samples: {num_surfaces:,}")

        # Draw parameters
        lnvol = self.rng.uniform(rg['LnVol'][0], rg['LnVol'][1], num_surfaces)
        beta = self.rng.uniform(rg['Beta'][0], rg['Beta'][1], num_surfaces)
        nu = self.rng.uniform(rg['Nu'][0], rg['Nu'][1], num_surfaces)
        rho = self.rng.uniform(rg['Rho'][0], rg['Rho'][1], num_surfaces)
        gamma = self.rng.uniform(rg['Gamma'][0], rg['Gamma'][1], num_surfaces)

        # Calculate prices per surface
        ts = []
        strikes = []
        fwds = []
        lnvols = []
        betas = []
        nus = []
        rhos = []
        gammas = []
        prices = []
        for j in range(num_surfaces):
            print(f"Surface generation number {j+1:,}/{num_surfaces:,}")
            expiries = self.rng.uniform(rg['Ttm'][0], rg['Ttm'][1], self.num_expiries)
            # Need to sort these expiries
            expiries = np.unique(expiries)
            fwd = self.rng.uniform(rg['F'][0], rg['F'][1], 1)[0]
            vol = lnvol[j]

            # Draw strikes
            ks = []
            for exp_idx in range(self.num_expiries):
                # Spread method
                # spread = self.rng.uniform(-300, 300, self.num_strikes)
                # k = fwd + spread / 10000.0
                # k = np.maximum(k, -shift + constants.BPS10)

                # Percentile method
                stdev = vol * np.sqrt(expiries[exp_idx])
                percentiles = self.rng.uniform(rg['K'][0], rg['K'][1], self.num_strikes)
                k = (fwd + shift) * np.exp(-0.5 * stdev * stdev + stdev * sp.norm.ppf(percentiles))
                k = k - shift

                ks.append(k)

            ks = np.asarray(ks)

            # Draw parameters
            params = {'LnVol': lnvol[j], 'Beta': beta[j], 'Nu': nu[j], 'Rho': rho[j],
                      'Gamma': gamma[j]}

            # Calculate prices
            price = self.price(expiries, ks, self.are_calls, fwd, params)

            # Flatten the results
            for exp_idx, expiry in enumerate(expiries):
                ts.extend([expiry] * self.num_strikes)
                strikes.extend(ks[exp_idx])
                fwds.extend([fwd] * self.num_strikes)
                lnvols.extend([lnvol[j]] * self.num_strikes)
                betas.extend([beta[j]] * self.num_strikes)
                nus.extend([nu[j]] * self.num_strikes)
                rhos.extend([rho[j]] * self.num_strikes)
                gammas.extend([gamma[j]] * self.num_strikes)
                prices.extend(price[exp_idx])

        # Put in dataframe
        df = pd.DataFrame({'Ttm': ts, 'K': strikes, 'F': fwds, 'LnVol': lnvols, 'Beta': betas,
                           'Nu': nus, 'Rho': rhos, 'Gamma': gammas, 'Price': prices})
        df.columns = ['Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Gamma', 'Price']

        return df

    def price(self, expiries, strikes, are_calls, fwd, parameters):
        shifted_k = strikes + self.shift
        shifted_f = fwd + self.shift
        prices = mczabr.price(expiries, shifted_k, are_calls, shifted_f, parameters,
                              self.num_mc, self.points_per_year)

        return prices

    def retrieve_datasets_no_shuffle(self, data_df):
        # Retrieve suitable data
        t = data_df.Ttm
        strike = data_df.K
        fwd = data_df.F
        lnvol = data_df.LnVol
        beta = data_df.Beta
        nu = data_df.Nu
        rho = data_df.Rho
        gamma = data_df.Gamma
        nvol = data_df.NVol

        # Extract input and output datasets
        x_set = np.column_stack((t, strike, fwd, lnvol, beta, nu, rho, gamma))
        num_samples = len(nvol)
        y_set = np.asarray(nvol)
        y_set = np.reshape(y_set, (num_samples, 1))

        return x_set, y_set

    def price_surface_mod(self, model, expiries, strikes, are_calls, fwd, parameters):
        # Retrieve parameters
        lnvol = parameters['LnVol']
        beta = parameters['Beta']
        nu = parameters['Nu']
        rho = parameters['Rho']
        gamma = parameters['Gamma']

        # Prepare learning model inputs
        num_expiries = expiries.shape[0]
        num_strikes = strikes.shape[1]
        num_points = num_expiries * num_strikes
        md_inputs = np.ones((num_points, 8))
        md_inputs[:, 0] = np.repeat(expiries, num_strikes)
        md_inputs[:, 1] = strikes.reshape(-1)
        md_inputs[:, 2] *= fwd
        md_inputs[:, 3] *= lnvol
        md_inputs[:, 4] *= beta
        md_inputs[:, 5] *= nu
        md_inputs[:, 6] *= rho
        md_inputs[:, 7] *= gamma

        # Flatten are_calls
        flat_types = np.asarray(are_calls).reshape(-1)

        # Price with learning model
        md_nvols = model.predict(md_inputs)
        md_prices = []
        for (point, vol, is_call) in zip(md_inputs, md_nvols, flat_types):
            expiry = point[0]
            strike = point[1]
            md_prices.append(bachelier.price(expiry, strike, is_call, fwd, vol))

        md_prices = np.asarray(md_prices)
        return md_prices.reshape(num_expiries, num_strikes)

# # Special class for Shifted ZABR with shift = 3%, for easier calling
# class McShiftedZabrGenerator(McZabrGenerator):
#     """ For calling convenience, derived from McZabrGenerator with shift at typical 3%. """
#     def __init__(self, num_expiries=15, num_strikes=10, num_mc=10000, points_per_year=10):
#         McZabrGenerator.__init__(self, 0.03, num_expiries, num_strikes, num_mc, points_per_year)


if __name__ == "__main__":
    # Test generation
    NUM_SAMPLES = 100 #100 * 1000
    MODEL_TYPE = 'McZABR'
    SHIFT = 0.03
    NUM_MC = 100 * 1000
    POINTS_PER_YEAR = 25
    SURFACE_SIZE = 50
    NUM_EXPIRIES = 10
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    project_folder = os.path.join(settings.WORKFOLDER, "stovol")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_tests.tsv")
    generator = McZabrGenerator(SHIFT, NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

    ranges = {'Ttm': [1.0 / 12.0, 5.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041],
              'LnVol': [0.05, 0.25], 'Beta': [0.49, 0.51], 'Nu': [0.20, 0.80], 'Rho': [-0.4, 0.4],
              'Gamma': [0.10, 0.9]}
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
