import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(42)

# Multiple charts
m = 100
x = 6 * rng.rand(m, 1) - 3
# print(x)
y = 0.5 * x**2 + x + 2 + 0.25 * np.random.randn(m, 1)
# print(y)
plot_axis = np.linspace(-3, 3, num=10)
plot_axis = [[x] for x in plot_axis]
# print(plot_axis)
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
plt.scatter(x, y, s=2, color='black', alpha=0.50, label='Sample')
plt.plot(plot_axis, y_lin, color='blue', label="Linear")
plt.plot(plot_axis, y_poly, color='red', label="Poly")
plt.title("Compare regressions")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()

# Contours
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x, y, z)
plt.show()

# Sub-plots
# plt.figure(figsize=(11, 4))
#
# plt.subplot(121)
# plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=3, r=1, C=5$", fontsize=18)
#
# plt.subplot(122)
# plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
# plot_dataset(x, y, [-1.5, 2.5, -1, 1.5])
# plt.title(r"$d=10, r=100, C=5$", fontsize=18)
#
# plt.show()
