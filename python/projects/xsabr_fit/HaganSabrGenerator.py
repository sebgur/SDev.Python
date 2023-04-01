import os
import numpy as np
import pandas as pd
from analytics.sabr import sabr_iv
import analytics.black as black
from analytics.SmileGenerator import SmileGenerator
import tools.FileManager as FileManager


# ################ ToDo ###################################################################################
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