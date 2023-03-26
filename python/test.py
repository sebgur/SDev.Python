import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import tensorflow as tf

x = [[1, 2], [3, 4]]
print(x)
a = tf.math.reduce_prod(x, axis=1, keepdims=True)
print(a)


# x_train = np.array([0, 1, 2, 3, 4, 5])
# a = 0.2
# b = 2.5
# c = -1.4
# y_train = [a * x**2 + b * x + c for x in x_train]
# x_train = x_train.reshape(-1, 1)
# y_train = np.array(y_train).reshape(-1, 1)
# print(y_train)
#
# poly_features = PolynomialFeatures(degree=2, include_bias=True)
# x_poly = poly_features.fit_transform(x_train)
# poly_reg = LinearRegression()
# poly_reg.fit(x_poly, y_train)
# print(poly_reg.intercept_)
# print(poly_reg.coef_)
# print(poly_reg.coef_[0][2])
# coeffs = [poly_reg.coef_[0][2], poly_reg.coef_[0][1], poly_reg.intercept_[0]]
# print(coeffs)


