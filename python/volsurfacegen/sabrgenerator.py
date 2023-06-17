""" Smile generator for classic and shifted Hagan SABR models """
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
import settings
from analytics import sabr
from analytics import black
from analytics import bachelier
from volsurfacegen.smilegenerator import SmileGenerator
from tools import filemanager
from tools import constants


class SabrGenerator(SmileGenerator):
    """ Base class for the classical SABR model with a generic shift value. The classical Hagan
      model (non-shifted) is the default case with shift = 0. """
    def __init__(self, shift=0.0, num_expiries=15, num_strikes=10, seed=42):
        SmileGenerator.__init__(self)
        self.shift = shift
        self.num_strikes = num_strikes
        self.num_expiries = num_expiries
        self.surface_size = self.num_expiries * self.num_strikes
        self.are_calls = [[self.is_call] * self.num_strikes] * self.num_expiries

        self.rng = np.random.RandomState(seed)
        if self.shift > 0.01:
            # shift is often 3% conservatively, but training from 3% seems excessive
            self.min_fwd = -0.01 + constants.BPS10
        else:
            self.min_fwd = -shift + constants.BPS10

        self.max_fwd = self.min_fwd + 0.05
        self.num_curve_parameters = 1
        self.num_vol_parameters = 4

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
        lnvol = self.rng.uniform(0.05, 0.50, num_surfaces)
        beta = self.rng.uniform(0.10, 0.90, num_surfaces)
        # beta = self.rng.uniform(0.45, 0.55, num_surfaces)
        nu = self.rng.uniform(0.10, 1.00, num_surfaces)
        rho = self.rng.uniform(-0.60, 0.60, num_surfaces)

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
            expiries = self.rng.uniform(1.0 / 12.0, 35.0, self.num_expiries)
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
            params = {'LnVol': lnvol[j], 'Beta': beta[j], 'Nu': nu[j], 'Rho': rho[j]}

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
                prices.extend(price[exp_idx])

        # Put in dataframe
        df = pd.DataFrame({'Ttm': ts, 'K': strikes, 'F': fwds, 'LnVol': lnvols, 'Beta': betas,
                           'Nu': nus, 'Rho': rhos, 'Price': prices})
        df.columns = ['Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Price']

        return df

    # def generate_samples_old(self, num_samples):
    #     shift = self.shift
    #     t = self.rng.uniform(1.0 / 12.0, 32.0, num_samples)
    #     fwd = self.rng.uniform(self.min_fwd, self.max_fwd, num_samples)
    #     lnvol = self.rng.uniform(0.05, 0.50, num_samples)

    #     # Draw strikes by spreads
    #     # spread = self.rng.uniform(-300, 300, num_samples)
    #     # strike = fwd + spread / 10000.0
    #     # strike = np.maximum(strike, -shift + constants.BPS10)

    #     # Draw strikes by percentiles
    #     stdev = lnvol * np.sqrt(t)
    #     percentiles = self.rng.uniform(0.01, 0.99, num_samples)
    #     strike = (fwd + shift) * np.exp(-0.5 * stdev * stdev + stdev * sp.norm.ppf(percentiles))
    #     strike = strike - shift

    #     # Draw parameters
    #     beta = self.rng.uniform(0.45, 0.55, num_samples)
    #     nu = self.rng.uniform(0.10, 1.00, num_samples)
    #     rho = self.rng.uniform(-0.60, 0.60, num_samples)
    #     params = {'LnVol': lnvol, 'Beta': beta, 'Nu': nu, 'Rho': rho}

    #     # Calculate prices
    #     price = self.price(t, strike, self.is_call, fwd, params)

    #     # Put in dataframe
    #     df = pd.DataFrame({'Ttm': t, 'K': strike, 'F': fwd, 'LnVol': lnvol, 'Beta': beta, 'Nu': nu,
    #                        'Rho': rho, 'Price': price})
    #     df.columns = ['Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Price']

    #     return df

    # def price(self, expiry, strike, is_call, fwd, parameters):
    #     shifted_k = strike + self.shift
    #     shifted_f = fwd + self.shift
    #     iv = sabr.implied_vol_vec(expiry, shifted_k, shifted_f, parameters)
    #     price = black.price(expiry, shifted_k, is_call, shifted_f, iv)
    #     return price

    def price(self, expiries, strikes, are_calls, fwd, parameters):
        expiries_ = np.asarray(expiries).reshape(-1, 1)
        shifted_k = strikes + self.shift
        shifted_f = fwd + self.shift
        prices = []
        for i, expiry in enumerate(expiries_):
            k_prices = []
            for j, sk in enumerate(shifted_k[i]):
                iv = sabr.implied_vol_vec(expiry, sk, shifted_f, parameters)
                price = black.price(expiry, sk, are_calls[i][j], shifted_f, iv)
                k_prices.append(price[0])
            prices.append(k_prices)

        return np.asarray(prices)

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
        nvol = data_df.NVol

        # Extract input and output datasets
        x_set = np.column_stack((t, strike, fwd, lnvol, beta, nu, rho))
        num_samples = len(nvol)
        y_set = np.asarray(nvol)
        y_set = np.reshape(y_set, (num_samples, 1))

        return x_set, y_set, data_df

    def price_surface_ref(self, expiries, strikes, are_calls, fwd, parameters):
        # are_calls = [is_call] * strikes.shape[1]
        # are_calls = [are_calls] * expiries.shape[0]
        ref_prices = self.price(expiries, strikes, are_calls, fwd, parameters)
        return ref_prices

    # def price_surface_ref_old(self, expiries, strikes, is_call, fwd, parameters):
    #     ref_prices = self.price(expiries, strikes, is_call, fwd, parameters)
    #     return ref_prices

    def price_surface_mod(self, model, expiries, strikes, are_calls, fwd, parameters):
        # Retrieve parameters
        lnvol = parameters['LnVol']
        beta = parameters['Beta']
        nu = parameters['Nu']
        rho = parameters['Rho']

        # Prepare learning model inputs
        num_expiries = expiries.shape[0]
        num_strikes = strikes.shape[1]
        num_points = num_expiries * num_strikes
        md_inputs = np.ones((num_points, 7))
        md_inputs[:, 0] = np.repeat(expiries, num_strikes)
        md_inputs[:, 1] = strikes.reshape(-1)
        md_inputs[:, 2] *= fwd
        md_inputs[:, 3] *= lnvol
        md_inputs[:, 4] *= beta
        md_inputs[:, 5] *= nu
        md_inputs[:, 6] *= rho

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

    def convert_strikes(self, expiries, strike_inputs, fwd, parameters, input_method='Strikes'):
        if input_method == 'Percentiles':
            lnvol = parameters['LnVol']
            stdev = lnvol * np.sqrt(expiries)
            sfwd = fwd + self.shift
            return sfwd * np.exp(-0.5 * stdev**2 + stdev * sp.norm.ppf(strike_inputs)) - self.shift
        else:
            return SmileGenerator.convert_strikes(expiries, strike_inputs, fwd, parameters,
                                                  input_method)


# Special class for Shifted SABR with shift = 3%, for easier calling
class ShiftedSabrGenerator(SabrGenerator):
    """ For calling convenience, derived from SabrGenerator with shift at typical 3%. """
    def __init__(self, num_expiries=15, num_strikes=10, seed=42):
        SabrGenerator.__init__(self, shift=0.03, num_expiries=num_expiries, num_strikes=num_strikes,
                               seed=seed)


if __name__ == "__main__":
    # Test generation
    NUM_SAMPLES = 10 #100 * 1000
    MODEL_TYPE = 'ShiftedSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "stovol")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_test.tsv")
    generator = ShiftedSabrGenerator()

    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df_ = generator.generate_samples(NUM_SAMPLES)
    print(data_df_)
    print("Cleansing data")
    data_df_ = generator.to_nvol(data_df_)
    print("Output to file: " + file)
    generator.to_file(data_df_, file)
    print("Complete!")

    # Test strike conversion
    # generator = ShiftedSabrGenerator()
    # EXPIRIES = np.asarray([0.5, 1.0, 5.0]).reshape(-1, 1)
    # STRIKE_INPUTS = np.asarray([[0.1, 0.9], [0.4, 0.6], [0.5, 0.5]])
    # FWD = 0.01
    # PARAMETERS = { 'LnVol': 0.2222, 'Beta': 0.3333, 'Nu': 0.4444, 'Rho': -0.5555 }
    # METHOD = 'Percentiles'
    # prices = generator.price_surface_mod(None, EXPIRIES, STRIKE_INPUTS, FWD, PARAMETERS, METHOD)
    # print(prices)
