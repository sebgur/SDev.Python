from tools.settings import apply_settings
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time

apply_settings()

x = tf.constant(5.0)
with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x
    dy_dx = gg.gradient(y, x)  # dy_dx = 2 * x
d2y_dx2 = g.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx2)

#################################################
# ######## Generate a set of collocation points
#################################################
# Set number of data points
N = 5

tspace = np.linspace(0.0, 1.0, N)
print(tspace)
tspace = 0.5
# tspace = np.ones(N) * 0.5
print(tspace)
xspace = np.linspace(1.0, 2.0, N)
T, X = np.meshgrid(tspace, xspace)
# print(T)
# print(X)
Tf = T.flatten()
Xf = X.flatten()
# print(Tf)
# print(Xf)
Xgrid = np.vstack([T.flatten(), X.flatten()]).T
print(Xgrid)

# t_chart = 0.01
# x_space = np.linspace(0.0, 1.0, N)
# points = [t_chart, x_space[:]]
# print(points)

