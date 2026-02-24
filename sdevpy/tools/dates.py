import datetime as dt
from dateutil.relativedelta import relativedelta


DATE_FORMAT = '%d-%b-%Y'
DATETIME_FORMAT = '%d-%b-%Y %H:%M:%S'
DATE_FILE_FORMAT = '%Y%m%d-%H%M%S'


def date_advance(base_date, days=0, months=0, years=0):
    return base_date + relativedelta(days=days, months=months, years=years)


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))
