""" Standard metrics and statistical functions """
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf

def mse(set1, set2):
    """ Mean Squared Error """
    return mean_squared_error(set1, set2)

def rmsew(set1, set2, weights):
    """ Root Mean Squared Error with weights """
    return np.sqrt(mean_squared_error(set1, set2, sample_weight=weights))

def rmse(set1, set2):
    """ Root Mean Squared Error """
    return np.sqrt(mean_squared_error(set1, set2))

def bps_rmse(y_true, y_ref):
    """ RMSE in bps """
    return 10000.0 * rmse(y_true, y_ref)

# Tensorflow versions
def tf_mse(y_true, y_pred):
    """ Mean Squared Error in tensorflow """
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def tf_rmse(y_true, y_pred):
    """ Root Mean Squared Error in tensorflow """
    return tf.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred)))

def tf_bps_rmse(y_true, y_ref):
    """ RMSE in bps in tensorflow """
    return 10000.0 * tf_rmse(y_true, y_ref)
