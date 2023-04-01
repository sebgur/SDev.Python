import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(set1, set2):
    return np.sqrt(mean_squared_error(set1, set2)) * 10000
