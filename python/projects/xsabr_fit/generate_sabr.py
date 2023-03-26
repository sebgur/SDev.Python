import numpy as np
import pandas as pd
from analytics.sabr import sabr_iv, sabr_iv2


nSamples = 10
beta = 0.5
shift = 0.0
print("Beta = " + str(beta))
print("Shift = " + str(shift))

print("Setting up random number generator...")
rngSeed = 42
rng = np.random.RandomState(rngSeed)

# Generate
print("Generating samples....")
# beta = rng.uniform(0.10, 0.80, (nSamples, 1))
nu = rng.uniform(0.20, 0.80, (nSamples, 1))
rho = rng.uniform(-0.40, 0.40, (nSamples, 1))
F = rng.uniform(0.02, 0.04, (nSamples, 1))
alpha = rng.uniform(0.05, 0.25, (nSamples, 1)) / F**(beta - 1.0)
K = rng.uniform(0.01, 0.06, (nSamples, 1))
T = rng.uniform(1.0 / 12.0, 5.0, (nSamples, 1))
Price = sabr_iv2(alpha, beta, nu, rho, F, K, T)

# Export to csv file
print("Exporting to tsv file...")
data = pd.DataFrame({'alpha': alpha[:, 0], 'beta': beta[:, 0], 'nu': nu[:, 0], 'rho': rho[:, 0],
                    'F': F[:, 0], 'K': K[:, 0], 'T': T[:, 0], 'Price': Price[:, 0]})

data = data.drop(data[data.Price > 1.5].index)  # Remove high vols
data.columns = ['alpha', 'beta', 'nu', 'rho', 'F', 'K', 'T', 'Price']
data.to_csv("outputs/Hagan_SABR_samples.csv", sep=',', index=False)
print("Complete!")
