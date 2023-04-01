from abc import ABC, abstractmethod


class SmileGenerator(ABC):
    @abstractmethod
    def generate_samples(self, num_samples, output_file=""):
        pass

    @abstractmethod
    def price(self, expiry, strike, parameters):
        pass
