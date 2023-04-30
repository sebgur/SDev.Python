""" Just to test things """
import tensorflow as tf
import numpy as np
from maths.metrics import rmse
# import projects.xsabr_fit.sabrgenerator as sabr


def custom_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred)))


test = np.ones((10, 1)) * 12.0
ref = np.ones((10, 1)) * 23.0
loss = custom_mean_squared_error(test, ref).numpy()
print(loss)
rmse = rmse(test, ref)
print(rmse)