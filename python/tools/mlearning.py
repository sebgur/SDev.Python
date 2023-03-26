import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
# from sklearn.externals import joblib
import tensorflow as tf
import random as rn
import logging
import os


# Prepare training, testing and validation sets, output scaler to file
def prepare_sets(train_percent, test_percent, n_features, data_file, scaler_file):

    if train_percent + test_percent > 1.0:
        return "Error: training and testing set sizes larger than total size of samples"

    # Read samples
    print("Reading sample file...")
    data = pd.read_csv(data_file, sep=',')

    # Split data into training and testing sets
    data_size = data.shape[0]
    train_size = int(data_size * train_percent)
    test_size = int(data_size * test_percent)
    print("Training size: " + str(train_size))
    print("Testing size: " + str(test_size))
    print("Validation size: " + str(data_size - train_size - test_size))

    x_train = data.iloc[0:train_size, 0:n_features]
    x_test = data.iloc[train_size:train_size + test_size, 0:n_features]
    x_val = data.iloc[train_size + test_size:, 0:n_features]

    y_train = np.array(data.iloc[0:train_size, n_features:])
    y_test = np.array(data.iloc[train_size:train_size + test_size, n_features:])
    y_val = np.array(data.iloc[train_size + test_size:, n_features:])

    # Scale and transform
    x_scaler = MinMaxScaler(feature_range=(0, 1), copy=False)
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)
    x_val = x_scaler.transform(x_val)

    joblib.dump(x_scaler, scaler_file)
    return x_train, x_test, x_val, y_train, y_test, y_val


# Initialize Keras for reproducible results
def init_keras():
    np.random.seed(42)
    rn.seed(21)
    session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # session_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    from keras import backend as kback
    tf.random.set_seed(2019)
    # tf.set_random_seed(2019)
    session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_config)
    # session = tf.Session(graph=tf.get_default_graph(), config=session_config)
    kback.set_session(session)


# Turn off tensorflow warnings
def turn_off_ts_warnings():
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import tensorflow as tf
