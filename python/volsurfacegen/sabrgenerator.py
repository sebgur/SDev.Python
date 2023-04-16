""" Smile generator for classic and shifted Hagan SABR models """
import os
import numpy as np
import pandas as pd
import settings
from analytics import sabr
from analytics import black
from analytics import bachelier
from analytics.smilegenerator import SmileGenerator
from tools import filemanager


BPS10 = 10.0 / 10000.0


class SabrGenerator(SmileGenerator):
    """ Base class for the classical SABR model with a generic shift value. The classical Hagan
      model (non-shifted) is the default case with shift = 0. """
    def __init__(self, shift=0.0):
        SmileGenerator.__init__(self)
        self.shift = shift
        seed = 42
        self.rng = np.random.RandomState(seed)
        if self.shift > 0.01:
            self.min_fwd = -0.01 + BPS10  # shift is often 3%, but training from 3% seems excessive
        else:
            self.min_fwd = -shift + BPS10

        self.max_fwd = self.min_fwd + 0.05
        self.num_curve_parameters = 1
        self.num_vol_parameters = 4

    def generate_samples(self, num_samples):
        shift = self.shift
        t = self.rng.uniform(1.0 / 12.0, 5.0, (num_samples, 1))
        fwd = self.rng.uniform(self.min_fwd, self.max_fwd, (num_samples, 1))
        # Draw strikes
        spread = self.rng.uniform(-300, 300, (num_samples, 1))
        strike = fwd + spread / 10000.0
        strike = np.maximum(strike, -shift + BPS10)

        # Draw parameters
        ln_vol = self.rng.uniform(0.05, 0.25, (num_samples, 1))
        beta = self.rng.uniform(0.49, 0.51, (num_samples, 1))
        # alpha = ln_vol * (np.abs(fwd + shift)) ** (1.0 - beta)
        nu = self.rng.uniform(0.20, 0.80, (num_samples, 1))
        rho = self.rng.uniform(-0.40, 0.40, (num_samples, 1))
        params = np.column_stack((ln_vol, beta, nu, rho))

        # Calculate prices
        # shifted_k = strike + shift
        # shifted_f = fwd + shift
        # iv = sabr.implied_vol_vec(t, shifted_k, shifted_f, params)
        # price = black.price(t, shifted_k, shifted_f, iv, self.is_call)
        price = self.price(t, strike, self.is_call, fwd, params)

        # Put in dataframe
        df = pd.DataFrame({'Ttm': t[:, 0], 'K': strike[:, 0], 'F': fwd[:, 0],
                           'LnVol': ln_vol[:, 0], 'Beta': beta[:, 0], 'Nu': nu[:, 0],
                           'Rho': rho[:, 0], 'Price': price[:, 0]})
        df.columns = ['Ttm', 'K', 'F', 'LnVol', 'Beta', 'Nu', 'Rho', 'Price']

        return df

    def price(self, expiry, strike, is_call, fwd, parameters):
        shifted_k = strike + self.shift
        shifted_f = fwd + self.shift
        iv = sabr.implied_vol_vec(expiry, shifted_k, shifted_f, parameters)
        price = black.price(expiry, shifted_k, shifted_f, iv, is_call)
        return price

    def retrieve_datasets(self, data_file):
        data_df = SmileGenerator.from_file(data_file)

        # Retrieve suitable data
        t = data_df.TTM
        fwd = data_df.F
        strike = data_df.K
        nvol = data_df.NVOL
        alpha = data_df.Alpha
        beta = data_df.Beta
        nu = data_df.Nu
        rho = data_df.Rho

        # Extract input and output datasets
        x_set = np.column_stack((t, strike, fwd, alpha, beta, nu, rho))
        num_samples = len(nvol)
        y_set = np.asarray(nvol)
        y_set = np.reshape(y_set, (num_samples, 1))

        return x_set, y_set, data_df

    def price_strike_ladder(self, model, spreads, fwd, parameters):
        strikes = []
        # Calculate strikes and exclude those below the shift
        for spread in spreads:
            strike = fwd + spread / 10000.0
            if strike > self.shift + BPS10:
                strikes.append(strike)

        num_points = len(strikes)
        # Retrieve parameters
        # shift = self.shift
        ln_vol = parameters['LnVol']
        beta = parameters['Beta']
        # alpha = parameters['LnVol'] * (fwd + shift)**(1.0 - beta)
        nu = parameters['Nu']
        rho = parameters['Rho']
        expiry = parameters['Ttm']

        # Prepare learning model inputs
        md_inputs = np.ones((num_points, 7))
        md_inputs[:, 0] *= expiry
        md_inputs[:, 1] *= fwd # ToDo: reorder here too for consistency
        md_inputs[:, 2] = strikes
        md_inputs[:, 3] *= ln_vol
        md_inputs[:, 4] *= beta
        md_inputs[:, 5] *= nu
        md_inputs[:, 6] *= rho

        # Price with learning model
        md_nvols = model.predict(md_inputs)
        md_prices = bachelier.price(expiry, strike, self.is_call, fwd, md_nvols)

        # Price with ref valuation
        rf_params = [ln_vol, beta, nu, rho]
        rf_prices = []
        for strike in strikes:
            rf_prices.append(self.price(expiry, strike, self.is_call, fwd, rf_params))

        return rf_prices, md_prices

# Special classe for Shifted SABR with shift = 3%, for easier calling
class ShiftedSabrGenerator(SabrGenerator):
    """ For calling convenience, derived from SabrGenerator with shift at typical 3%. """
    def __init__(self):
        SabrGenerator.__init__(self, 0.03)


if __name__ == "__main__":
    print("<><><><> MODULE TEST <><><><>")
    NUM_SAMPLES = 100 * 1000
    MODEL_TYPE = 'ShiftedSABR'
    output_folder = os.path.join(settings.WORKFOLDER, "XSABRsamples")
    filemanager.check_directory(output_folder)
    file = os.path.join(output_folder, MODEL_TYPE + "_samples.tsv")
    generator = ShiftedSabrGenerator()

    print("Generating " + str(NUM_SAMPLES) + " samples")
    data_df_ = generator.generate_samples(NUM_SAMPLES)
    print("Cleansing data")
    data_df_ = generator.to_nvol(data_df_)
    print("Output to file: " + file)
    generator.to_file(data_df_, file)
    print("Complete!")
    print("<><><><><><><><><><><><><><>")

# Dump former generate_sabr_vec
# import numpy as np
# import pandas as pd
# from tools.sabr import sabr_iv
# from scipy.stats import norm
#
# print("Setting up random number generator...")
# n_samples = 2000000
# rngSeed = 42
# rng = np.random.RandomState(rngSeed)
#
# # Generate
# print("Generating samples....")
# sigma = rng.uniform(0.05, 0.25, (n_samples, 1))
# beta = rng.uniform(0.10, 0.80, (n_samples, 1))
# nu = rng.uniform(0.20, 0.80, (n_samples, 1))
# rho = rng.uniform(-0.40, 0.40, (n_samples, 1))
# F = rng.uniform(0.02, 0.04, (n_samples, 1))
# alpha = sigma / F**(beta - 1.0)
# T = rng.uniform(1.0 / 12.0, 5.0, (n_samples, 1))
#
# # Generate fixed strikes
# p = np.array([0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99])
# c = norm.ppf(p)
# n_strikes = p.size
# atm = sabr_iv(alpha, beta, nu, rho, F, F, T)
# stdev = atm * np.sqrt(T)
# K = np.ndarray(shape=(n_samples, n_strikes))
# for i in range(n_strikes):
#     K[:, i] = F[:, 0] * np.exp(-0.5 * stdev[:, 0]**2 + stdev[:, 0] * c[i])
#
# Price = np.ndarray(shape=(n_samples, n_strikes))
# for i in range(n_strikes):
#     Price[:, i] = sabr_iv(alpha[:, 0], beta[:, 0], nu[:, 0], rho[:, 0], F[:, 0], K[:, i], T[:, 0])
#
# # Export to csv file
# print("Exporting to csv file...")
# n_fixed = 6  # alpha, beta, nu, rho, F, T
# n_fields = n_fixed + n_strikes
# data = np.ndarray(shape=(n_samples, n_fields))
# data[:, 0] = alpha[:, 0]
# data[:, 1] = beta[:, 0]
# data[:, 2] = nu[:, 0]
# data[:, 3] = rho[:, 0]
# data[:, 4] = F[:, 0]
# data[:, 5] = T[:, 0]
# data_names = ['alpha', 'beta', 'nu', 'rho', 'F', 'T']
# for i in range(n_strikes):
#     data[:, n_fixed + i] = Price[:, i]
#     data_names.append(p[i])
#
# df = pd.DataFrame(data)
# df.columns = data_names
# df.to_csv("outputs/Hagan_SABR_vec_samples.csv", sep=',', index=False)
#
# print("Complete!")
