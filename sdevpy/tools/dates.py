import datetime as dt
from dateutil.relativedelta import relativedelta


DATE_FORMAT = '%d-%b-%Y'
DATETIME_FORMAT = '%d-%b-%Y %H:%M:%S'
DATE_FILE_FORMAT = '%Y%m%d-%H%M%S'


def list_dates(start_date, end_date, cdr=None):
    """ List dates between start and end, including both if they are business days. """
    dates = []
    date = start_date if is_business_day(start_date, cdr) else next_business_day(start_date, cdr)
    while date <= end_date:
        dates.append(date)
        date = next_business_day(date, cdr)

    return dates


def is_business_day(date, cdr=None):
    return date.weekday() >= 5


def next_business_day(date, cdr=None):
    next_date = date_advance(date, days=1)
    if is_business_day(next_date, cdr):
        return next_date
    else:
        return next_business_day(next_date)


# ToDo: rename to just 'advance'
def date_advance(base_date, days=0, months=0, years=0):
    """ Advancing by days, months and years with no calendar or convention considerations """
    return base_date + relativedelta(days=days, months=months, years=years)


# class Calendar:
#     def __init__(self):
#         pass


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))
