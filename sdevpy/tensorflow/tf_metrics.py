""" Standard metrics and statistical functions in Tensorflow.
    Development has stopped. Planning to move to PyTorch.
"""
import tensorflow as tf

def tf_mse(y_true, y_pred): # pragma: no cov
    """ Mean Squared Error in tensorflow """
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def tf_rmse(y_true, y_pred): # pragma: no cov
    """ Root Mean Squared Error in tensorflow """
    return tf.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred)))

def tf_bps_rmse(y_true, y_ref): # pragma: no cov
    """ RMSE in bps in tensorflow """
    return 10000.0 * tf_rmse(y_true, y_ref)
