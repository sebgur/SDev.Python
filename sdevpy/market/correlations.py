from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
from sdevpy.utilities import dates as dts


def get_correlations(names: list[str], date: dt.datetime, **kwargs):
    """ Retrieve correlation matrix for given names """
    # print(names)
    # Load data from file
    file = data_file(date, **kwargs)
    data_df = pd.read_csv(file, index_col=0)
    data_series = data_df.iloc[:, 0]

    # Retrieve data for pairs (upper triangle)
    n_names = len(names)
    corr_matrix = np.zeros(shape=(n_names, n_names))
    for i in range(len(names)):
        name1 = names[i]
        for j in range(len(names)):
            if j > i:
                name2 = names[j]
                key = f"{name1}-{name2}"
                if key in data_series:
                    corr_matrix[i, j] = data_series[key]
                else:
                    key = f"{name2}-{name1}"
                    if key in data_series:
                        corr_matrix[i, j] = data_series[key]
                    else:
                        raise KeyError(f"Name pair {key} not found in correlation file {file}")

    # Complete matrix
    for i in range(len(names)):
        for j in range(len(names)):
            if j == i:
                corr_matrix[i, j] = 1.0
            elif j < i:
                corr_matrix[i, j] = corr_matrix[j, i]

    # Make matrix
    # corr_matrix = np.array([[1.0, 0.5, 0.1],
    #                  [0.5, 1.0, 0.1],
    #                  [0.1, 0.1, 1.0]])
    return corr_matrix


def add_correlations(date: dt.datetime, names1: list[str], names2: list[str], values: list[float], **kwargs):
    """ Append new correlations at end of correlation file """
    if not len(names1) == len(names2) == len(values):
        raise ValueError("Incompatible sizes in new correlation data")

    file = data_file(date, **kwargs)
    with open(file, 'a') as f:
        for name1, name2, value in zip(names1, names2, values, strict=True):
            f.write(f"{name1}-{name2},{value}\n")


def data_file(date: dt.datetime, **kwargs):
    folder = kwargs.get('folder', test_data_folder())
    file = Path(folder) / (date.strftime(dts.DATE_FILE_FORMAT) + ".csv")
    return file


def test_data_folder():
    folder = Path(__file__).parent.parent.parent / "datasets" / "marketdata" / "correlations"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


if __name__ == "__main__":
    date = dt.datetime(2025, 12, 15)
    corr = get_correlations(['ABC', 'KLM', 'XYZ'], date)
    print(corr)
    # add_correlations(date, ['dddd', 'eeee'], ['cccc', 'ffff'], [0.2, 0.3])
