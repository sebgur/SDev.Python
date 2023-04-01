from abc import ABC, abstractmethod


class SmileGenerator(ABC):
    def __init__(self):
        self.num_curve_parameters = 0
        self.num_vol_parameters = 0

    @abstractmethod
    def generate_samples(self, num_samples, output_file=""):
        pass

    @abstractmethod
    def price(self, expiry, strike, parameters):
        pass

    def num_parameters(self):
        return self.num_curve_parameters + self.num_vol_parameters
