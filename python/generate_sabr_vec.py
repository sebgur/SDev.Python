import numpy as np
import pandas as pd
from tools.sabr import sabr_iv
from scipy.stats import norm

print("Setting up random number generator...")
n_samples = 2000000
rngSeed = 42
rng = np.random.RandomState(rngSeed)

# Generate
print("Generating samples....")
sigma = rng.uniform(0.05, 0.25, (n_samples, 1))
beta = rng.uniform(0.10, 0.80, (n_samples, 1))
nu = rng.uniform(0.20, 0.80, (n_samples, 1))
rho = rng.uniform(-0.40, 0.40, (n_samples, 1))
F = rng.uniform(0.02, 0.04, (n_samples, 1))
alpha = sigma / F**(beta - 1.0)
T = rng.uniform(1.0 / 12.0, 5.0, (n_samples, 1))

# Generate fixed strikes
p = np.array([0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99])
c = norm.ppf(p)
n_strikes = p.size
atm = sabr_iv(alpha, beta, nu, rho, F, F, T)
stdev = atm * np.sqrt(T)
K = np.ndarray(shape=(n_samples, n_strikes))
for i in range(n_strikes):
    K[:, i] = F[:, 0] * np.exp(-0.5 * stdev[:, 0]**2 + stdev[:, 0] * c[i])

Price = np.ndarray(shape=(n_samples, n_strikes))
for i in range(n_strikes):
    Price[:, i] = sabr_iv(alpha[:, 0], beta[:, 0], nu[:, 0], rho[:, 0], F[:, 0], K[:, i], T[:, 0])

# Export to csv file
print("Exporting to csv file...")
n_fixed = 6  # alpha, beta, nu, rho, F, T
n_fields = n_fixed + n_strikes
data = np.ndarray(shape=(n_samples, n_fields))
data[:, 0] = alpha[:, 0]
data[:, 1] = beta[:, 0]
data[:, 2] = nu[:, 0]
data[:, 3] = rho[:, 0]
data[:, 4] = F[:, 0]
data[:, 5] = T[:, 0]
data_names = ['alpha', 'beta', 'nu', 'rho', 'F', 'T']
for i in range(n_strikes):
    data[:, n_fixed + i] = Price[:, i]
    data_names.append(p[i])

df = pd.DataFrame(data)
df.columns = data_names
df.to_csv("outputs/Hagan_SABR_vec_samples.csv", sep=',', index=False)

print("Complete!")
