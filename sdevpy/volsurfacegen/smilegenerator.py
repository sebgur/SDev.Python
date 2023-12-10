""" Base framework for smile generation """
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import scipy.stats as sp
from sdevpy.analytics import bachelier
from sdevpy.machinelearning import datasets


class SmileGenerator(ABC):
    """ Base class for smile generation """
    def __init__(self, shift=0.0, num_expiries=15, num_strikes=10, seed=42):
        self.is_call = False  # Use put options by default
        self.shift = shift
        self.num_strikes = num_strikes
        self.num_expiries = num_expiries
        self.surface_size = self.num_expiries * self.num_strikes
        self.are_calls = [[self.is_call] * self.num_strikes] * self.num_expiries
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def generate_samples(self, num_samples, rg):
        """ Generate a sample of expiries, strikes, relevant parameters and option prices """

    @abstractmethod
    def generate_samples_inverse(self, num_samples, rg, spreads):
        """ Generate an inverse sample of expiries, strikes, relevant parameters and option prices """

    @abstractmethod
    def price(self, expiries, strikes, are_calls, fwd, parameters):
        """ Calculate option price under the specified model and its parameters """
        # ToDo: couldn't we identify this with price_surface_ref? Ideally rename as
        # price_options_ref given that now we have price_straddles_ref

    @abstractmethod
    def price_straddles_ref(self, expiries, strikes, fwd, parameters):
        """ Calculate straddle prices under the specified model and its parameters """

    @abstractmethod
    def price_straddles_mod(self, model, expiries, strikes, fwd, mkt_prices):
        """ Calculate straddle prices for given parameters using the learning model """
        raise NotImplementedError("Straddle pricing with model not implemented yet")

    # #### Retrieve direct datasets ####
    def retrieve_datasets(self, data_file, shuffle=False):
        """ Retrieve dataset stored in tsv file """
        data_df = SmileGenerator.from_file(data_file, shuffle)
        x_set, y_set = self.retrieve_datasets_from_df(data_df, False)
        return x_set, y_set, data_df

    def retrieve_datasets_from_df(self, data_df, shuffle=False):
        """ Retrieve dataset from dataframe """
        if shuffle:
            data_df = datasets.shuffle_dataframe(data_df)

        return self.retrieve_datasets_no_shuffle(data_df)

    @abstractmethod
    def retrieve_datasets_no_shuffle(self, data_df):
        """ Retrieve dataset from dataframe without shuffling """

    # #### Retrieve inverse datasets ####
    def retrieve_inverse_datasets(self, data_file, shuffle=False):
        """ Retrieve inverse dataset stored in tsv file """
        data_df = SmileGenerator.from_file(data_file, shuffle)
        x_set, y_set = self.retrieve_inverse_datasets_from_df(data_df, False)
        return x_set, y_set, data_df

    def retrieve_inverse_datasets_from_df(self, data_df, shuffle=False):
        """ Retrieve inverse dataset from dataframe """
        if shuffle:
            data_df = datasets.shuffle_dataframe(data_df)

        return self.retrieve_inverse_datasets_no_shuffle(data_df)

    @abstractmethod
    def retrieve_inverse_datasets_no_shuffle(self, data_df):
        """ Retrieve inverse dataset from dataframe without shuffling """
        raise NotImplementedError("Method not implemented yet for chosen model")

    def price_surface_ref(self, expiries, strikes, are_calls, fwd, parameters):
        """ Calculate a surface of prices for given parameters using the reference model """
        return self.price(expiries, strikes, are_calls, fwd, parameters)

    @abstractmethod
    def price_surface_mod(self, model, expiries, strikes, are_calls, fwd, parameters):
        """ Calculate a surface of prices for given parameters using the learning model """

    def convert_strikes(self, expiries, strike_inputs, fwd, parameters, input_method='Strikes'):
        """ Convert strike inputs into absolute strikes using the strike input_method """
        # pylint: disable=unused-argument
        if input_method == 'Strikes':
            strikes = strike_inputs
        elif input_method == 'Spreads':
            strikes = fwd + strike_inputs / 10000.0
        elif input_method == 'Percentiles':
            if 'LnVol' in parameters:
                lnvol = parameters['LnVol']
                stdev = lnvol * np.sqrt(expiries)
                sfwd = fwd + self.shift
                N = sp.norm
                return sfwd * np.exp(-0.5 * stdev**2 + stdev * N.ppf(strike_inputs)) - self.shift
            else:
                raise RuntimeError("Lognormal vol parameter not provided")
        else:
            raise ValueError("Invalid strike input method: " + input_method)

        return strikes

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
        num_print = 10000
        num_batches = int(num_samples / num_print) + 1
        batch_id = 0
        for i in range(num_samples):
            if i % num_print == 0:
                batch_id = batch_id + 1
                print(f"Converting to normal vol, batch {batch_id:,} out of {num_batches:,}")
            try:
                nvol.append(bachelier.implied_vol(t[i], strike[i], self.is_call, fwd[i], price[i]))
            except (Exception,):
                nvol.append(-9999)

        np.seterr(divide='warn')  # Set back to warning

        data_df['NVol'] = nvol
        # data_df['BSVol'] = bsvol

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


# if __name__ == "__main__":
    # Test reading from remote location and calculating prices and vols
