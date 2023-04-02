import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


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
