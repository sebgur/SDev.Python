""" Various utilities for software versions and so on """
import struct
import pandas as pd
import numpy as np
import numpy.typing as npt
from collections.abc import Iterable
import random
from sdevpy.maths.constants import EPS


def isequal(a: npt.ArrayLike, b: npt.ArrayLike, tol: float=EPS) -> bool:
    """ Check if equal, up to given tolerance for float """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return False

    if np.issubdtype(a_arr.dtype, np.floating) or np.issubdtype(b_arr.dtype, np.floating):
        return bool(np.allclose(a_arr, b_arr, atol=tol, rtol=0.0))

    return bool(np.array_equal(a_arr, b_arr))


def isiterable(x):
    """ Beware that this answers True for strings """
    return isinstance(x, Iterable)


def pd_read_xls(xls_file, col_name, index_col):
    """ Read Excel file and pick column of data and index """
    xls = pd.ExcelFile(xls_file)
    return xls.parse(col_name, index_col=index_col)


def print_python_bit_version():
    """ Prints number of bits of the installed python version (32 or 64) """
    print(struct.calcsize("P") * 8)


def hash():
    return str(rand_n_digits(10))


def rand_n_digits(n):
    """ Random integer with n digits """
    start = 10**(n - 1)
    end = 10**n
    return random.randrange(start, end)


if __name__ == '__main__':
    print(EPS)
