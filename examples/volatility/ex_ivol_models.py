""" Create a number of implied volatility models and view their smiles """
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sdevpy.volatility.impliedvol.models import biexp, svi, cubicvol, vsvi, gsvi


#### Select valuation dates, terms and percentiles ####
valdate = dt.datetime(2025, 12, 15)
terms = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
percents = [0.10, 0.25, 0.50, 0.75, 0.90]
base_vol = 0.25


## BiExp model ##
print("Running BiExp model")
params = [0.25, 0.30, 0.28, 1.0, 1.5, 0.0]
x = np.linspace(-5, 5, 100)
v = biexp.biexp(x, *params)
plt.plot(x, v)
plt.title('BiExp smile')
plt.show()

expiries, fwds, strikes, vols = biexp.generate_sample_data(valdate, terms, base_vol, percents)
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        exp_idx = n_cols * i + j
        ax.plot(strikes[exp_idx], vols[exp_idx], color='red')
        ax.set_title(expiries[exp_idx])
        ax.set_xlabel('strike')
        ax.set_ylabel('vol')

fig.suptitle('BiExp Vols', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


## SVI model ##
print("Running SVI model")
# One chart example
t = 1/365
alnv = 0.25
a = alnv**2 * t # a > 0
b = 0.0 # a / np.log(2) # b > 0
rho = 0.0 # -1 < rho < 1
m = 0.0 # No constraints
sigma = 0.0 # 0.5 / np.sqrt(t) # > 0
params = [a, b, rho, m, sigma]

k = np.linspace(0.2, 3.0, 100)
x = np.log(k)

vol = svi.svi_formula(t, x, params)
plt.plot(x, vol)
plt.title('SVI smile')
plt.show()

# Vectorization
times = np.asarray([t])
times = np.full_like(k, times)
print(f"times: {times.shape}")
params = [np.full_like(k, np.asarray([a])), np.full_like(k, np.asarray([b])), np.full_like(k, np.asarray([rho])),
          np.full_like(k, np.asarray([m])), np.full_like(k, np.asarray([sigma]))]
vols = svi.svi_formula(times, x, params)
print(f"shape: {vols.shape}")


## CubicVol model ##
print("Running CubicVol model")
t = 1.5
atm, skew, kurt, vl, vr = 0.25, 0.1, 0.25, 0.30, 0.27

ms = np.linspace(0.2, 4.0, 100)
lms = np.log(ms)
cubic_vols = cubicvol.cubicvol(t, lms, atm, skew, kurt, vl, vr)
print(cubic_vols)


## vSVI model ##
print("Running vSVI model")
expiries, fwds, strikes, vols = vsvi.generate_sample_data(valdate, terms, base_vol, percents)
n_rows, n_cols = 3, 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        exp_idx = n_cols * i + j
        ax.plot(strikes[exp_idx], vols[exp_idx], color='red')
        ax.set_title(expiries[exp_idx])
        ax.set_xlabel('strike')
        ax.set_ylabel('vol')

fig.suptitle('vSVI Vols', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


## gSVI model ##
print("Running gSVI model")
a, b, rho, m, sigma = base_vol, 0.1, -0.25, 0.0, 0.25 # a, b, rho, m, sigma
mx = np.asarray([0.5, 1.0, 2.0]) # Moneyness
log_m = np.log(mx) # Log-moneyness

test = gsvi.gsvi_formula(log_m, [a, b, rho, m, sigma])
print(test)
