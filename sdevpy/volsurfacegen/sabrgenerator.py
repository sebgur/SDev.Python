""" Smile generator for classic and shifted Hagan SABR models """
import os
import numpy as np
import pandas as pd
import scipy.stats as sp
from sdevpy import settings
from sdevpy.analytics import sabr
from sdevpy.analytics import black
from sdevpy.analytics import bachelier
from sdevpy.volsurfacegen.smilegenerator import SmileGenerator
from sdevpy.tools import filemanager


class SabrGenerator(SmileGenerator):
    """ Base class for the classical SABR model. The classical Hagan model (non-shifted) is the
        default case with shift = 0. """
    def __init__(self, shift=0.0, num_expiries=15, num_strikes=10, seed=42):
        SmileGenerator.__init__(self, shift, num_expiries, num_strikes, seed)

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

    def retrieve_datasets_no_shuffle(self, data_df):
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

        return x_set, y_set


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

        # Get normal vol from learning model
        md_nvols = model.predict(md_inputs)

        # Price with Bachelier
        md_prices = []
        for (point, vol, is_call) in zip(md_inputs, md_nvols, flat_types):
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


if __name__ == "__main__":
    # Test generation
    NUM_SAMPLES = 150 #100 * 1000
    MODEL_TYPE = 'SABR'
    SHIFT = 0.03
    project_folder = os.path.join(settings.WORKFOLDER, "stovol")
    data_folder = os.path.join(project_folder, "samples")
    filemanager.check_directory(data_folder)
    file = os.path.join(data_folder, MODEL_TYPE + "_samples_test.tsv")
    generator = SabrGenerator(SHIFT)

    ranges = {'Ttm': [1.0 / 12.0, 35.0], 'K': [0.01, 0.99], 'F': [-0.009, 0.041],
              'LnVol': [0.05, 0.50], 'Beta': [0.1, 0.9], 'Nu': [0.1, 1.0], 'Rho': [-0.6, 0.6]}
    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df_ = generator.generate_samples(NUM_SAMPLES, ranges)
    print(data_df_)
    print("Cleansing data")
    data_df_ = generator.to_nvol(data_df_)
    print("Output to file: " + file)
    generator.to_file(data_df_, file)
    print("Complete!")

    # Test price ref
    # NUM_STRIKES = 100
    # PARAMS = { 'LnVol': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25 }
    # FWD = 0.028

    # # Any number of expiries can be calculated, but for optimum display choose no more than 6
    # EXPIRIES = np.asarray([0.125, 0.250, 0.5, 1.00, 2.0, 5.0]).reshape(-1, 1)
    # # EXPIRIES = np.asarray([0.25, 0.50, 1.0, 5.00, 10.0, 30.0]).reshape(-1, 1)
    # NUM_EXPIRIES = EXPIRIES.shape[0]
    # METHOD = 'Percentiles'
    # PERCENTS = np.linspace(0.01, 0.99, num=NUM_STRIKES)
    # PERCENTS = np.asarray([PERCENTS] * NUM_EXPIRIES)

    # STRIKES = generator.convert_strikes(EXPIRIES, PERCENTS, FWD, PARAMS, METHOD)
    # ARE_CALLS = [[False] * NUM_STRIKES] * NUM_EXPIRIES # All puts

    # ref_prices = generator.price_surface_ref(EXPIRIES, STRIKES, ARE_CALLS, FWD, PARAMS)

    # Test strike conversion
    # generator = ShiftedSabrGenerator()
    # EXPIRIES = np.asarray([0.5, 1.0, 5.0]).reshape(-1, 1)
    # STRIKE_INPUTS = np.asarray([[0.1, 0.9], [0.4, 0.6], [0.5, 0.5]])
    # FWD = 0.01
    # PARAMETERS = { 'LnVol': 0.2222, 'Beta': 0.3333, 'Nu': 0.4444, 'Rho': -0.5555 }
    # METHOD = 'Percentiles'
    # prices = generator.price_surface_mod(None, EXPIRIES, STRIKE_INPUTS, FWD, PARAMETERS, METHOD)
    # print(prices)
