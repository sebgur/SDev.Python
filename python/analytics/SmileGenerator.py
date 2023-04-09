""" Base framework for smile generation """
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import analytics.bachelier as bachelier


class SmileGenerator(ABC):
    """ Base class for smile generation """
    def __init__(self):
        self.num_curve_parameters = 0
        self.num_vol_parameters = 0

    @abstractmethod
    def generate_samples(self, num_samples):
        """ Abstract base class """
        # pass

    @abstractmethod
    def price(self, expiry, strike, parameters):
        """ Abstract base class """
        # pass

    @abstractmethod
    def retrieve_datasets(self, data_file):
        """ Abstract base class """
        # pass

    def num_parameters(self):
        """ Total number of parameters (curve + vol) """
        return self.num_curve_parameters + self.num_vol_parameters

    @staticmethod
    def cleanse(data_df, cleanse=True, min_vol=0.0001, max_vol=0.1):
        """ Calculate normal implied vol and remove errors. Further remove points that are not
            in the given min/max range """
        # Calculate normal vols
        # print(np.geterr())
        np.seterr(divide='raise')  # To catch errors and warnings
        t = data_df.TTM
        fwd = data_df.F
        strike = data_df.K
        price = data_df.Price
        nvol = []
        num_samples = t.shape[0]
        for i in range(num_samples):
            try:
                nvol.append(bachelier.impliedvol(t[i], fwd[i], strike[i], price[i], is_call=False))
            except (Exception,):
                nvol.append(-9999)

        np.seterr(divide='warn')  # Set back to warning

        data_df['IV'] = nvol

        # Remove out of range
        if cleanse:
            data_df = data_df.drop(data_df[data_df.IV > max_vol].index)
            data_df = data_df.drop(data_df[data_df.IV < min_vol].index)

        return data_df

    @staticmethod
    def from_file(data_file):
        """ Creating dataframe from tsv file """
        data_df = pd.read_csv(data_file, sep='\t')
        return data_df

    @staticmethod
    def to_file(data_df, output_file):
        """ Dumping dataframe to tsv file """
        data_df.to_csv(output_file, sep='\t', index=False)
