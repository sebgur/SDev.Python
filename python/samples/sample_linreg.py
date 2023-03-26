import os
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# 'r' in front of string signals 'raw string' i.e. no more lynter warning due to escape charaters
path = r"C:\Users\sebgu\OneDrive\Work\ML\Datasets\lifesat"

def prepare_country_stats(oecd_bli_, gdp_per_capita_):
    oecd_bli_ = oecd_bli_[oecd_bli_["INEQUALITY"] == "TOT"]  # Select entries satisfying condition
    # print(oecd_bli_.shape)  # Shows the dataframe's (number of rows, number of columns)
    oecd_bli_ = oecd_bli_.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita_.rename(columns={"2015": "GDP per capita"}, inplace=True)  # Rename columns
    gdp_per_capita_.set_index("Country", inplace=True)  # Replace existing index
    full_country_stats = pd.merge(left=oecd_bli_, right=gdp_per_capita_, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


# Load the data
oecd_bli = pd.read_csv(os.path.join(path, "oecd_bli_2015.csv"), thousands=',')
gdp_per_capita = pd.read_csv(os.path.join(path, "gdp_per_capita.csv"), thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")
# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
print(country_stats.shape)
print(country_stats)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Select a linear model
lin_reg_model = LinearRegression()

# Train the model
lin_reg_model.fit(X, y)

# Make a prediction
X_new = [[22587]]
print()
print('Prediction at x = ' + str(X_new[0][0]) + ': ' + str(lin_reg_model.predict(X_new)[0][0]))

# Plot
# country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
t0, t1 = lin_reg_model.intercept_[0], lin_reg_model.coef_[0][0]
X = np.linspace(0, 60000, 1000)
# plt.plot(X, t0 + t1 * X, "b")
# plt.show()

# Polynomial regression (just add features and do linear regression)
rng = np.random.RandomState(42)
m = 100
x = 6 * rng.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + 0.9 * np.random.randn(m, 1)
plot_axis = np.linspace(-3, 3, num=100)
plot_axis = [[x] for x in plot_axis]
# Linear regression
lin_reg = LinearRegression()
lin_reg.fit(x, y)
y_lin = lin_reg.predict(plot_axis)
# Polynomial regression
poly_features = PolynomialFeatures(degree=2, include_bias=True)
x_poly = poly_features.fit_transform(x)
poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)
plot_axis_poly = poly_features.transform(plot_axis)
y_poly = poly_reg.predict(plot_axis_poly)

# Plot
# plt.scatter(x, y, s=2, color='black', alpha=0.50)
# plt.plot(plot_axis, y_lin, color='blue')
# plt.plot(plot_axis, y_poly, color='red')
# plt.show()

# ## Regularization ##
y = 0.5 * x + 2 + 0.25 * np.random.randn(m, 1)
poly_features = PolynomialFeatures(degree=20, include_bias=True)
x_poly = poly_features.fit_transform(x)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_poly)
plot_axis_poly = poly_features.transform(plot_axis)
plot_axis_poly = scaler.transform(plot_axis_poly)
# Ridge regularization with Cholesky solver
ridge_reg = Ridge(alpha=0.5, solver="cholesky")
ridge_reg.fit(x_scaled, y)
y_ridge = ridge_reg.predict(plot_axis_poly)
# Ridge regularization with Stochastic Average GD solver (should be the same as SGD with l2-reg)
ridge_reg = Ridge(alpha=0.0001, solver="sag")
ridge_reg.fit(x_scaled, y)
y_ridge_sag = ridge_reg.predict(plot_axis_poly)
# SGD with Ridge (l2) regularization
sgd_reg = SGDRegressor(penalty="l2", alpha=0.5)
sgd_reg.fit(x_scaled, y)
y_sgd = sgd_reg.predict(plot_axis_poly)
# Lasso regularization
lasso_reg = Lasso(alpha=0.0001)
lasso_reg.fit(x_scaled, y)
y_lasso = lasso_reg.predict(plot_axis_poly)
# Elastic Net regularization
elnet_reg = ElasticNet(alpha=0.0001, l1_ratio=0.5)
elnet_reg.fit(x_scaled, y)
y_elnet = elnet_reg.predict(plot_axis_poly)
# Plot
plt.scatter(x, y, s=2, color='black', alpha=0.50, label='Sample')
plt.plot(plot_axis, y_ridge, color='blue', label='Ridge-Cholesky')
plt.plot(plot_axis, y_ridge_sag, color='green', label='Ridge-SAG')
plt.plot(plot_axis, y_sgd, color='red', label='SGD-Ridge')
plt.plot(plot_axis, y_lasso, color='yellow', label='Lasso')
plt.plot(plot_axis, y_elnet, color='orange', label='Elastic Net')
plt.legend(loc='best')
# plt.plot(plot_axis, y_poly, color='red')
# plt.plot(plot_axis, y_ridge, color='green')
plt.show()

# Early stopping, sample
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate="constant", eta0=0.0005)
minimum_val_error = float("inf")
best_epoch, best_model, x_train, y_train, x_val, y_val = None, None, None, None, None, None
for epoch in range(1000):
    sgd_reg.fit(x_train, y_train)  # Continue where we left off thanks to warm_start = True
    y_val_predict = sgd_reg.predict(x_val)
    val_error = mean_squared_error(y_val_predict, y_val)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = clone(sgd_reg)
