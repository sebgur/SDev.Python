""" Base framework for smile generation """
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from analytics import bachelier


class SmileGenerator(ABC):
    """ Base class for smile generation """
    def __init__(self):
        self.num_curve_parameters = 0
        self.num_vol_parameters = 0
        self.is_call = False  # Use put options by default

    @abstractmethod
    def generate_samples(self, num_samples):
        """ Generate a sample of expiries, strikes, relevant parameters and option prices """

    @abstractmethod
    def price(self, expiry, strike, is_call, fwd, parameters):
        """ Calculate option price under the specified model and its parameters """

    @abstractmethod
    def retrieve_datasets(self, data_file, shuffle=False):
        """ Retrieve dataset stored in tsv file """

    # @abstractmethod
    # def price_strike_ladder(self, model, expiry, spreads, fwd, parameters):
    #     """ Calculate prices for a ladder of strikes for given parameters """

    @abstractmethod
    def price_surface_ref(self, expiries, strikes, is_call, fwd, parameters):
        """ Calculate a surface of prices for given parameters using the generating model """

    @abstractmethod
    def price_surface_mod(self, model, expiries, strikes, is_call, fwd, parameters):
        """ Calculate a surface of prices for given parameters using the learning model """

    def convert_strikes(self, expiries, strike_inputs, fwd, parameters, input_method='Strikes'):
        """ Convert strike inputs into absolute strikes using the strike input_method """
        if input_method == 'Strikes':
            strikes = strike_inputs
        elif input_method == 'Spreads':
            strikes = fwd + strike_inputs / 10000.0
        else:
            raise ValueError("Invalid strike input method: " + input_method)
        
        return strikes

    def num_parameters(self):
        """ Total number of parameters (curve + vol) """
        return self.num_curve_parameters + self.num_vol_parameters

    def to_nvol(self, data_df, cleanse=True, min_vol=0.0001, max_vol=0.1):
        """ Calculate normal implied vol and remove errors. Further remove points that are not
            in the given min/max range """
        # Calculate normal vols
        np.seterr(divide='raise')  # To catch errors and warnings
        t = data_df.Ttm
        fwd = data_df.F
        strike = data_df.K
        price = data_df.Price
        nvol = []
        num_samples = t.shape[0]
        for i in range(num_samples):
            try:
                nvol.append(bachelier.implied_vol(t[i], strike[i], self.is_call, fwd[i], price[i]))
            except (Exception,):
                nvol.append(-9999)

        np.seterr(divide='warn')  # Set back to warning

        data_df['NVol'] = nvol

        # Remove out of range
        if cleanse:
            data_df = data_df.drop(data_df[data_df.NVol > max_vol].index)
            data_df = data_df.drop(data_df[data_df.NVol < min_vol].index)

        return data_df

    def target_is_call(self):
        """ True if the fit target is call options, False if puts """
        return self.is_call

    @staticmethod
    def from_file(data_file, shuffle=False):
        """ Creating dataframe from tsv file """
        data_df = pd.read_csv(data_file, sep='\t')
        if shuffle:
            data_df = data_df.sample(frac=1)

        return data_df

    @staticmethod
    def to_file(data_df, output_file):
        """ Dumping dataframe to tsv file """
        data_df.to_csv(output_file, sep='\t', index=False)
