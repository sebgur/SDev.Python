""" Smile generator for classic and shifted Hagan SABR models """
import os
import numpy as np
import pandas as pd
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
    def __init__(self, shift=0.0):
        SmileGenerator.__init__(self)
        self.shift = shift
        seed = 42
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
        t = self.rng.uniform(1.0 / 12.0, 5.0, num_samples)
        fwd = self.rng.uniform(self.min_fwd, self.max_fwd, num_samples)
        # Draw strikes
        spread = self.rng.uniform(-300, 300, num_samples)
        strike = fwd + spread / 10000.0
        strike = np.maximum(strike, -shift + constants.BPS10)

        # Draw parameters
        lnvol = self.rng.uniform(0.05, 0.25, num_samples)
        beta = self.rng.uniform(0.49, 0.51, num_samples)
        nu = self.rng.uniform(0.20, 0.80, num_samples)
        rho = self.rng.uniform(-0.40, 0.40, num_samples)
        params = {'LnVol': lnvol, 'Beta': beta, 'Nu': nu, 'Rho': rho}

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
        iv = sabr.implied_vol_vec(expiry, shifted_k, shifted_f, parameters)
        price = black.price(expiry, shifted_k, is_call, shifted_f, iv)
        return price

    def retrieve_datasets(self, data_file):
        data_df = SmileGenerator.from_file(data_file)

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

    def price_strike_ladder(self, model, expiry, spreads, fwd, parameters):
        strikes = []
        # Calculate strikes and exclude those below the shift
        for spread in spreads:
            strike = fwd + spread / 10000.0
            if strike > self.shift + constants.BPS10:
                strikes.append(strike)

        num_points = len(strikes)
        # Retrieve parameters
        lnvol = parameters['LnVol']
        beta = parameters['Beta']
        nu = parameters['Nu']
        rho = parameters['Rho']

        # Prepare learning model inputs
        md_inputs = np.ones((num_points, 7))
        md_inputs[:, 0] *= expiry
        md_inputs[:, 1] = strikes
        md_inputs[:, 2] *= fwd
        md_inputs[:, 3] *= lnvol
        md_inputs[:, 4] *= beta
        md_inputs[:, 5] *= nu
        md_inputs[:, 6] *= rho

        # Price with learning model
        md_nvols = model.predict(md_inputs)
        md_prices = bachelier.price(expiry, strike, self.is_call, fwd, md_nvols)

        # Price with ref valuation
        rf_params = [lnvol, beta, nu, rho]
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
    NUM_SAMPLES = 10 #100 * 1000
    MODEL_TYPE = 'ShiftedSABR'
    project_folder = os.path.join(settings.WORKFOLDER, "xsabr")
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
