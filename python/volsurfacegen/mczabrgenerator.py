""" Smile generator for ZABR model using Monte-Carlo to calculate option prices """
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
import settings
from analytics import mczabr
# from analytics import black
from analytics import bachelier
from volsurfacegen.sabrgenerator import SabrGenerator
from volsurfacegen.smilegenerator import SmileGenerator
from tools import filemanager
from tools import timer


class McZabrGenerator(SabrGenerator):
    """ ZABR model with a generic shift value, using Monte-Carlo to calculate option prices. """
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
        num_surfaces = int(num_samples / self.surface_size)# + 1)
        print(f"Number of surfaces/parameter samples: {num_surfaces:,}")

        # Draw parameters
        lnvol = self.rng.uniform(0.05, 0.25, num_surfaces)
        beta = self.rng.uniform(0.49, 0.51, num_surfaces)
        nu = self.rng.uniform(0.20, 0.80, num_surfaces)
        rho = self.rng.uniform(-0.40, 0.40, num_surfaces)
        gamma = self.rng.uniform(0.10, 1.4, num_surfaces)

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
            expiries = self.rng.uniform(1.0 / 12.0, 5.0, self.num_expiries)
            # Need to sort these expiries
            expiries = np.unique(expiries)
            fwd = self.rng.uniform(self.min_fwd, self.max_fwd, 1)[0]
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
                percentiles = self.rng.uniform(0.01, 0.99, self.num_strikes)
                k = (fwd + shift) * np.exp(-0.5 * stdev * stdev + stdev * sp.norm.ppf(percentiles))
                k = k - shift

                ks.append(k)

            ks = np.asarray(ks)

            # Draw parameters
            params = {'LnVol': lnvol[j], 'Beta': beta[j], 'Nu': nu[j], 'Rho': rho[j], 'Gamma': gamma[j]}

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

    def price(self, expiry, strike, is_call, fwd, parameters):
        shifted_k = strike + self.shift
        shifted_f = fwd + self.shift
        prices = mczabr.price(expiry, shifted_k, is_call, shifted_f, parameters,
                              self.num_mc, self.points_per_year)

        return prices

    def retrieve_datasets(self, data_file, shuffle=False):
        data_df = SmileGenerator.from_file(data_file, shuffle)

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

        return x_set, y_set, data_df

    # def price_surface_ref(self, expiries, strikes, is_call, fwd, parameters):
    #     are_calls = [is_call] * strikes.shape[1]
    #     are_calls = [are_calls] * expiries.shape[0]
    #     ref_prices = self.price(expiries, strikes, are_calls, fwd, parameters)
    #     return ref_prices

    def price_surface_mod(self, model, expiries, strikes, is_call, fwd, parameters):
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

        # Price with learning model
        md_nvols = model.predict(md_inputs)
        md_prices = []
        for (point, vol) in zip(md_inputs, md_nvols):
            expiry = point[0]
            strike = point[1]
            md_prices.append(bachelier.price(expiry, strike, is_call, fwd, vol))

        md_prices = np.asarray(md_prices)
        return md_prices.reshape(num_expiries, num_strikes)

    # def convert_strikes(self, expiries, strike_inputs, fwd, parameters, input_method='Strikes'):
    #     if input_method == 'Percentiles':
    #         lnvol = parameters['LnVol']
    #         stdev = lnvol * np.sqrt(expiries)
    #         sfwd = fwd + self.shift
    #         return sfwd * np.exp(-0.5 * stdev**2 + stdev * sp.norm.ppf(strike_inputs)) - self.shift
    #     else:
    #         return SmileGenerator.convert_strikes(expiries, strike_inputs, fwd, parameters,
    #                                               input_method)

# Special class for Shifted ZABR with shift = 3%, for easier calling
class McShiftedZabrGenerator(McZabrGenerator):
    """ For calling convenience, derived from McZabrGenerator with shift at typical 3%. """
    def __init__(self, num_expiries=15, num_strikes=10, num_mc=10000, points_per_year=10):
        McZabrGenerator.__init__(self, 0.03, num_expiries, num_strikes, num_mc, points_per_year)


if __name__ == "__main__":
    NUM_SAMPLES = 100 #100 * 1000
    NUM_MC = 100 * 1000
    POINTS_PER_YEAR = 25
    SURFACE_SIZE = 50
    NUM_EXPIRIES = 10
    NUM_STRIKES = int(SURFACE_SIZE / NUM_EXPIRIES)
    MODEL_TYPE = 'McShiftedZABR'
    project_folder = os.path.join(settings.WORKFOLDER, "stovol")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples.tsv")
    generator = McShiftedZabrGenerator(NUM_EXPIRIES, NUM_STRIKES, NUM_MC, POINTS_PER_YEAR)

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
