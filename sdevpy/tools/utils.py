""" Various utilities for software versions and so on """
import struct
import pandas as pd
from collections.abc import Iterable


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

if __name__ == '__main__':
    print("Hello")
