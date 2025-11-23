""" Standard metrics and statistical functions in Tensorflow """
import tensorflow as tf

def tf_mse(y_true, y_pred):
    """ Mean Squared Error in tensorflow """
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def tf_rmse(y_true, y_pred):
    """ Root Mean Squared Error in tensorflow """
    return tf.sqrt(tf.math.reduce_mean(tf.square(y_true - y_pred)))

def tf_bps_rmse(y_true, y_ref):
    """ RMSE in bps in tensorflow """
    return 10000.0 * tf_rmse(y_true, y_ref)
