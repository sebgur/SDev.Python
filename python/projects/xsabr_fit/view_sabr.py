import tools.sabr as sabr

model_name = 'Hagan_SABR_vec'
scaler_file = 'outputs/' + model_name + '_scaler.h5'
model_file = 'outputs/' + model_name + '_model.h5'

# Calculate range of strikes
def calculate_strikes(alpha, beta, nu, rho, f, t, percentiles):
    atm = sabr_iv(alpha, beta, nu, rho, f, f, t)
    stdev = atm * np.sqrt(t)
    confidence = norm.ppf(percentiles)
    k = f * np.exp(-0.5 * stdev**2 + stdev * confidence)
    return k


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


sabr.view_sabr_smiles_vec(model_file, scaler_file)
