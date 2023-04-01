import os
import numpy as np
import pandas as pd
from analytics.sabr import sabr_iv
import analytics.black as black


# ################ ToDo ###################################################################################
# Create XsabrGenerator base class
# Make Hagan shifted sabr derive from class
# Compare correctness against C#
# Implement generic cleansing and conversion from prices
# Implement generic training


nSamples = 10
output_folder = ""
shift = 0.0
print("Shift = " + str(shift))

print("Setting up random number generator...")
rngSeed = 42
rng = np.random.RandomState(rngSeed)

# Generate
print("Generating samples....")
t = rng.uniform(1.0 / 12.0, 5.0, (nSamples, 1))
spread = rng.uniform(-300, 300, (nSamples, 1))
fwd = rng.uniform(0.02, 0.04, (nSamples, 1))
strike = fwd + spread / 10000.0
beta = rng.uniform(0.10, 0.80, (nSamples, 1))
alpha = rng.uniform(0.05, 0.25, (nSamples, 1)) / (fwd + shift)**(beta - 1.0)
nu = rng.uniform(0.20, 0.80, (nSamples, 1))
rho = rng.uniform(-0.40, 0.40, (nSamples, 1))
iv = sabr_iv(t, strike + shift, fwd + shift, alpha, beta, nu, rho)
price = black.price(t, strike + shift, fwd + shift, iv, is_call=False)

# Export to csv file
print("Exporting to tsv file...")
data = pd.DataFrame({'T': t[:, 0], 'K': strike[:, 0], 'F': fwd[:, 0], 'Alpha': alpha[:, 0],
                     'Beta': beta[:, 0], 'Nu': nu[:, 0], 'Rho': rho[:, 0], 'Price': iv[:, 0]})

# data = data.drop(data[data.IV > 1.5].index)  # Remove high vols
data.columns = ['T', 'K', 'F', 'Alpha', 'Beta', 'Nu', 'Rho', 'Price']
file = os.path.join(output_folder, "Hagan_SABR_samples.tsv")
data.to_csv(file, sep='\t', index=False)
print("Complete!")
