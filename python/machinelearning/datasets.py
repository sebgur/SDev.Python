""" Dataset preparation for training """
import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib


def prepare_sets(inputs, outputs, train_percent):
    """ Split input and output datasets into training and test datasets split
        according to specified percentage to put in the training set """
    data_size = inputs.shape[0]
    if outputs.shape[0] != data_size:
        raise RuntimeError("Incompatible sizes between inputs and outputs")

    train_size = int(data_size * train_percent)

    train_inputs = inputs[0:train_size]
    train_outputs = outputs[0:train_size]
    test_inputs = inputs[train_size:data_size]
    test_outputs = outputs[train_size:data_size]

    return train_inputs, train_outputs, test_inputs, test_outputs


if __name__ == "__main__":
    INPUTS = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    OUTPUTS = np.asarray([[1], [2], [3], [4], [5]])
    train_x, train_y, test_x, test_y = prepare_sets(INPUTS, OUTPUTS, 0.80)
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)
