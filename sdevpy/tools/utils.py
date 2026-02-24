""" Various utilities for software versions and so on """
import struct
import pandas as pd
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass
from collections.abc import Iterable


@dataclass
class DateTimeSpan:
    days: int = 0
    months: int = 0
    years: int = 1


def isiterable(x):
    """ Beware that this answers True for strings """
    return isinstance(x, Iterable)


def date_advance(base_date, span=DateTimeSpan()):
    return base_date - relativedelta(days=span.days, months=span.months, years=span.years)


def pd_read_xls(xls_file, col_name, index_col):
    """ Read Excel file and pick column of data and index """
    xls = pd.ExcelFile(xls_file)
    return xls.parse(col_name, index_col=index_col)     


def print_python_bit_version():
    """ Prints number of bits of the installed python version (32 or 64) """
    print(struct.calcsize("P") * 8)

if __name__ == '__main__':
    import datetime as dt
    today = dt.date.today()
    hist_window = DateTimeSpan(years=1)
    hist_start = date_advance(today, hist_window)
    # hist_start = today - relativedelta(days=hist_window[0], months=hist_window[1], years=hist_window[2])

    print(today.strftime('%d-%b-%Y'))
    print(hist_start.strftime('%d-%b-%Y'))
