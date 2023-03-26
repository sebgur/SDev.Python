import sys
# sys.path.insert(0, ".") # When using VSCode
import numpy as np
import matplotlib.pyplot as plt
import tools.utils as utils
from keras.models import load_model
from sklearn.externals import joblib
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import CubicSpline


# Intermediate function for Hagan's SABR
def chi(z, rho):
    eps_sabr = 0.0001

    tmp1 = np.sqrt(1.0 - 2.0 * rho * z + z * z)
    np.where(abs(z) < eps_sabr, z, z)

    zz = np.where((tmp1 + z - rho > 0.0),
                  np.log((tmp1 + z - rho) / (1.0 - rho)), np.log((1.0 + rho) / (tmp1 - (z - rho))))

    return zz


# Hagan's SABR formula
def sabr_iv2(alpha, beta, nu, rho, f, k, t):
    eps_sabr = 0.0001
    v = (f * k) ** ((1.0 - beta) / 2.0)
    log_fk = np.log(f / k)
    tmp1 = nu * nu * (2.0 - 3.0 * rho * rho) / 24.0
    tmp1 += rho * beta * nu * alpha / (v * 4.0)
    tmp1 += (1.0 - beta) * (1.0 - beta) * alpha * alpha / ((v ** 2) * 24.0)
    tmp1 = alpha * (1.0 + (tmp1 * t))
    tmp2 = v * (1.0 + np.power(log_fk * (1.0 - beta), 2.0) / 24.0 + np.power(log_fk * (1.0 - beta), 4.0) / 1920.0)
    z = nu / alpha * v * log_fk
    chi_z = chi(z, rho)
    vol = np.where(abs(f - k) > eps_sabr, tmp1 / tmp2 * z / chi_z, tmp1 / np.power(f, 1.0 - beta))
    return vol


# Hagan's SABR formula with ATM limit handling
def sabr_iv(alpha, beta, nu, rho, f, k, t):
    big_a = np.power(f * k, 1.0 - beta)
    sqrt_a = np.sqrt(big_a)
    log_m = np.log(f / k)

    z = (nu / alpha) * sqrt_a * log_m
    m_epsilon = 1e-15
    # if z * z > 10.0 * m_epsilon:
    #     multiplier = z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    # else:
    #     multiplier = 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0

    # The below is for vectors but where() evaluates both so gives warning when going on the wrong side
    multiplier = np.where(z * z > 10.0 * m_epsilon,
                          z / np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho)),
                          1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0)

    big_c = np.power((1 - beta) * log_m, 2)
    big_d = sqrt_a * (1.0 + big_c / 24.0 + np.power(big_c, 2) / 1920.0)
    d = 1.0 + t * (np.power((1 - beta) * alpha, 2) / (24.0 * big_a) + 0.25 * rho * beta * nu * alpha / sqrt_a
                   + (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0))

    return (alpha / big_d) * multiplier * d


# Calculate range of strikes
def calculate_strikes(alpha, beta, nu, rho, f, t, percentiles):
    atm = sabr_iv(alpha, beta, nu, rho, f, f, t)
    stdev = atm * np.sqrt(t)
    confidence = norm.ppf(percentiles)
    k = f * np.exp(-0.5 * stdev**2 + stdev * confidence)
    return k


# sigma1 = 0.15
# #alpha1 = 0.05163535285867484
# beta1 = 0.1 # 0.7655000144869414
# nu1 = 0.8 #0.6391963650868431
# rho1 = -0.30 #0.07892678735762926
# f1 = 0.02 #0.02312037280884873
# t1 = 1 / 12 #0.8503063916529964
# alpha1 = sigma1 / f1**(beta1 - 1)
# # vol = sabr_iv(alpha1, beta1, nu1, rho1, f1, f1, t1)
# percentiles1 = np.linspace(0.01, 0.99, num=1000)
# k1 = calculate_strikes(alpha1, beta1, nu1, rho1, f1, t1, percentiles1)
# #sabr_vector(alpha1, beta1, nu1, rho1, f1, t1, k1)
# old = sabr_iv(alpha1, beta1, nu1, rho1, f1, k1, t1)
# new = sabr_vector(alpha1, beta1, nu1, rho1, f1, t1, k1)
# plt.ioff()
# plt.plot(k1, old, color='blue', label='old')
# plt.plot(k1, new, color='red', label='new')
# plt.show()


# Plot one smile, displaying corresponding SABR parameters
def plot_sabr_smile(K, smile_cf, smile_nn, alpha, beta, nu, rho, f, T, rmse):
    plt.title('T=%.2f' % T + ', F=%.2f' %(f*100) + ', alpha=%.2f' %(alpha * 100) + ', beta=%.2f' %beta + ',\n' +
              'nu=%.2f' %(nu*100) + ', rho=%.2f' %(rho * 100) + ', rmse=%.1f' %rmse + 'bps')
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.plot(K, smile_cf, color='blue', label='Closed-Form')
    plt.plot(K, smile_nn, color='red', label='NN')
    plt.legend(loc='upper right')


# Create one SABR IV plot against Hagan
def check_sabr_smile(trained_model, scaler, sigma, beta, nu, rho, f, t):
    n_points = 100
    k = np.linspace(0.01, 0.08, num=n_points)
    alpha = sigma / f**(beta - 1.0)
    smile_cf = sabr_iv(alpha, beta, nu, rho, f, k, t)
    test_points = np.ndarray(shape=(n_points, 7))
    for i in range(n_points):
        test_points[i, 0] = alpha
        test_points[i, 1] = beta
        test_points[i, 2] = nu
        test_points[i, 3] = rho
        test_points[i, 4] = f
        test_points[i, 5] = k[i]
        test_points[i, 6] = t

    scaler.transform(test_points)
    smile_nn = trained_model.predict(test_points)
    rmse = utils.rmse(smile_cf, smile_nn)
    plot_sabr_smile(k, smile_cf, smile_nn, alpha, beta, nu, rho, f, t, rmse)


# Create multiple SABR IV plots against Hagan
def view_sabr_smiles(model_file, scaler_file):
    trained_model = load_model(model_file)
    scaler = joblib.load(scaler_file)

    plt.ioff()
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.40)
    plt.subplot(3, 2, 1)
    check_sabr_smile(trained_model, scaler, 0.10, 0.10, 0.80, -0.30, 0.020, 1/12)
    plt.subplot(3, 2, 2)
    check_sabr_smile(trained_model, scaler, 0.12, 0.20, 0.70, -0.20, 0.022, 0.50)
    plt.subplot(3, 2, 3)
    check_sabr_smile(trained_model, scaler, 0.15, 0.30, 0.60, -0.10, 0.025, 1)
    plt.subplot(3, 2, 4)
    check_sabr_smile(trained_model, scaler, 0.15, 0.40, 0.50, +0.00, 0.025, 2)
    plt.subplot(3, 2, 5)
    check_sabr_smile(trained_model, scaler, 0.20, 0.50, 0.40, +0.10, 0.027, 5)
    plt.subplot(3, 2, 6)
    check_sabr_smile(trained_model, scaler, 0.25, 0.60, 0.30, +0.20, 0.030, 10)
    plt.show()


# Cleanse SABR sample data
def cleanse_sabr(sample_file):
    data = pd.read_csv(sample_file, sep=',')
    n_original = data.shape[0]
    print("Number of original items: " + str(n_original))
    data = data.drop(data[data.Price > 1.5].index)  # Remove high vols
    data = data.drop(data[data.Price < 0.01].index)  # Remove low vols
    n_removed = n_original - data.shape[0]
    print("Number of removed items: " + str(n_removed))
    data.to_csv(sample_file, sep=',', index=False)


# Cleanse SABR sample data
# def cleanse_sabr_vector(samples):
#    new_samples = [sample for sample in samples if 0.01 < samples[7] < 1.5]
#    return new_samples


# Interpolate model output with natural cubic spline
def interpolate_model(in_strikes, alpha, beta, nu, rho, f, t, model, scaler):
    percentiles = np.array([0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99])
    strikes = calculate_strikes(alpha, beta, nu, rho, f, t, percentiles)
    points = np.ndarray(shape=(1, 6))
    points[0] = [alpha, beta, nu, rho, f, t]
    scaler.transform(points)
    vols = model.predict(points)
    interpolation = CubicSpline(strikes, vols[0], bc_type='natural')
    return interpolation(in_strikes)


# Create one SABR IV plot against Hagan
def check_sabr_smile_vec(model, scaler, sigma, beta, nu, rho, f, t):
    n_points = 100
    alpha = sigma / f**(beta - 1.0)
    p = np.linspace(0.001, 0.999, num=n_points)
    k = calculate_strikes(alpha, beta, nu, rho, f, t, p)

    smile_cf = sabr_iv(alpha, beta, nu, rho, f, k, t)
    smile_nn = interpolate_model(k, alpha, beta, nu, rho, f, t, model, scaler)
    rmse = utils.rmse(smile_cf, smile_nn)
    plot_sabr_smile(k, smile_cf, smile_nn, alpha, beta, nu, rho, f, t, rmse)


# View smiles for vector format
def view_sabr_smiles_vec(model_file, scaler_file):
    model = load_model(model_file)
    scaler = joblib.load(scaler_file)

    plt.ioff()
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace=0.40)
    plt.subplot(3, 2, 1)
    check_sabr_smile_vec(model, scaler, 0.10, 0.10, 0.80, -0.30, 0.020, 1/12)
    plt.subplot(3, 2, 2)
    check_sabr_smile_vec(model, scaler, 0.12, 0.20, 0.70, -0.20, 0.022, 0.50)
    plt.subplot(3, 2, 3)
    check_sabr_smile_vec(model, scaler, 0.15, 0.30, 0.60, -0.10, 0.025, 1)
    plt.subplot(3, 2, 4)
    check_sabr_smile_vec(model, scaler, 0.15, 0.40, 0.50, +0.00, 0.025, 2)
    plt.subplot(3, 2, 5)
    check_sabr_smile_vec(model, scaler, 0.20, 0.50, 0.40, +0.10, 0.027, 5)
    plt.subplot(3, 2, 6)
    check_sabr_smile_vec(model, scaler, 0.25, 0.60, 0.30, +0.20, 0.030, 10)
    plt.show()
