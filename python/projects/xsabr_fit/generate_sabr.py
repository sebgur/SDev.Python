import os
import numpy as np
import pandas as pd
from analytics.sabr import sabr_iv
import analytics.black as black
from analytics.SmileGenerator import SmileGenerator
import tools.FileManager as FileManager


# ################ ToDo ###################################################################################
# Compare correctness against C#
# Implement generic cleansing and conversion from prices
# Implement generic training

class HaganSabrGenerator(SmileGenerator):
    def __init__(self, shift=0.0):
        self.shift = shift
        seed = 42
        self.rng = np.random.RandomState(seed)

    def generate_samples(self, num_samples, output_file=""):
        t = self.rng.uniform(1.0 / 12.0, 5.0, (num_samples, 1))
        spread = self.rng.uniform(-300, 300, (num_samples, 1))
        fwd = self.rng.uniform(0.02, 0.04, (num_samples, 1))
        strike = fwd + spread / 10000.0
        beta = self.rng.uniform(0.10, 0.80, (num_samples, 1))
        alpha = self.rng.uniform(0.05, 0.25, (num_samples, 1)) / (fwd + self.shift) ** (beta - 1.0)
        nu = self.rng.uniform(0.20, 0.80, (num_samples, 1))
        rho = self.rng.uniform(-0.40, 0.40, (num_samples, 1))
        iv = sabr_iv(t, strike + self.shift, fwd + self.shift, alpha, beta, nu, rho)
        price = black.price(t, strike + self.shift, fwd + self.shift, iv, is_call=False)

        # Put in dataframe
        data = pd.DataFrame({'T': t[:, 0], 'K': strike[:, 0], 'F': fwd[:, 0], 'Alpha': alpha[:, 0],
                             'Beta': beta[:, 0], 'Nu': nu[:, 0], 'Rho': rho[:, 0], 'Price': price[:, 0]})
        data.columns = ['T', 'K', 'F', 'Alpha', 'Beta', 'Nu', 'Rho', 'Price']

        # Dump to file
        data.to_csv(output_file, sep='\t', index=False)

        return data

    def price(self, expiry, strike, parameters):
        fwd = parameters[0]
        alpha = parameters[1]
        beta = parameters[2]
        nu = parameters[3]
        rho = parameters[4]
        iv = sabr_iv(expiry, strike + self.shift, fwd + self.shift, alpha, beta, nu, rho)
        price = black.price(expiry, strike + self.shift, fwd + self.shift, iv, is_call=False)
        return price


# Test
def test():
    shift = 0.03  # Shift often taken to 3%
    num_samples = 10
    print("Shift = " + str(shift))
    output_folder = r"W:\Data\XSABRsamples"
    FileManager.check_directory(output_folder)
    # data = data.drop(data[data.IV > 1.5].index)  # Remove high vols
    file = os.path.join(output_folder, "Hagan_SABR_samples.tsv")
    generator = HaganSabrGenerator(shift)
    # Generate samples
    print("Generating " + str(num_samples) + " samples")
    generator.generate_samples(num_samples, file)
    print("Complete!")


test()
