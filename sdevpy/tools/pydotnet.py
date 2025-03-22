from sdevpy.settings import *
import datetime as dt
import numpy as np
import pandas as pd
from System import *

def to_oadate(cs_date):
    return int(cs_date.ToOADate())


def to_csdate(py_date):
    return DateTime(py_date.year, py_date.month, py_date.day)


def to_csdatetime(py_date):
    return DateTime(py_date.year, py_date.month, py_date.day,
                    py_date.hour, py_date.minute, py_date.second,
                    int(py_date.microsecond / 1000))


def to_date(cs_date):
    return dt.datetime(cs_date.Year, cs_date.Month, cs_date.Day)


def to_nparray(cs_matrix):
    """ Convert C# object[,] to numpy matrix """
    try:
        type = cs_matrix.GetType().Name
        if type == 'Object[,]' or type == 'String[,]':
            n_rows = cs_matrix.GetLength(0)
            n_cols = cs_matrix.GetLength(1)
            res = []
            res.extend(cs_matrix)
            return np.asarray(res).reshape(n_rows, n_cols)
        else:
            raise TypeError("C# not of expected type Object[,]: " + type)
    except Exception as e:
        raise TypeError("Could not convert C# object to numpay array: " + repr(e))


def to_dataframe(cs_matrix):
    np_array = to_nparray(cs_matrix)
    return pd.DataFrame(data=np_array[1:, 1:], columns=np_array[0, 1:])


def to_list(cs_matrix):
    """ Convert C# object[,] to list if object is effectively 1-dimensional """
    np_array = to_nparray(cs_matrix)
    shape = np_array.shape
    p_list = []
    if shape[0] == 1:
        p_list = [np_array[0, i] for i in range(shape[1])]
    elif shape[1] == 1:
        p_list = [np_array[i, 0] for i in range(shape[0])]
    else:
        raise RuntimeError("Numpay array is not a list")
    
    return p_list


if __name__ == "__main__":
    print("Hi again " + USERNAME)

    iv = 0.20
    fwd = 0.04
    strike = 0.03
    maturity = 2.5
    is_call = True
    price = wf.xlBachelier.sdBachelierPrice(fwd, strike, iv, is_call)
    print(price)