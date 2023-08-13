""" Dataset preparation for training """
import os
import numpy as np
import pandas as pd
from sdevpy.tools import filemanager


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

def retrieve_data(folder, num_samples, shuffle=True, sep='\t', export_file=""):
    """ Use all files in folder to create a dataset, shuffle its data,
        extract num_samples from it and return dataframe """

    # Set extension
    if sep == '\t':
        extension = ".tsv"
    elif sep == ',':
        extension = ".csv"
    else:
        raise RuntimeError("Unknown text file separation")

    # Merge content of folder in single dataframe
    files = filemanager.list_files(folder, [extension])
    df = pd.DataFrame()
    for f in files:
        new_df = pd.read_csv(os.path.join(folder, f), sep=sep)
        df = pd.concat([df, new_df])

    if shuffle:
        df = df.sample(frac=1)

    # Clip num_samples
    df = clip_dataframe(df, num_samples)

    # If export_file is not empty, export to file
    if export_file != "":
        df.to_csv(export_file, sep=sep, index=False)

    return df

def clip_dataframe(df, size):
    """ Clip dataframe beyond specified size"""
    df_size = len(df.index)
    if df_size <= size:
        return df
    else:
        return df.iloc[range(0, size)]
    
def shuffle_dataframe(df, frac=1):
    """ Shuffle dataframe """
    df = df.sample(frac=frac)
    return df


if __name__ == "__main__":
    INPUTS = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    OUTPUTS = np.asarray([[1], [2], [3], [4], [5]])
    train_x, train_y, test_x, test_y = prepare_sets(INPUTS, OUTPUTS, 0.80)
    print(train_x)
    print(train_y)
    print(test_x)
    print(test_y)

    # Test merging
    FOLDER = r"W:\Sandbox\SDev.Python\samples\SABR"
    NUM_SAMPLES = 50000
    DATA_FILE = r"W:\Sandbox\SDev.Python\samples\SABR.tsv"
    DATA_DF = retrieve_data(FOLDER, NUM_SAMPLES, export_file=DATA_FILE, shuffle=True)
    print(DATA_DF)
