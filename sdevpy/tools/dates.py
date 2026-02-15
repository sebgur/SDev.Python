import datetime as dt


DATE_FORMAT = '%d-%b-%Y'
DATETIME_FORMAT = '%d-%b-%Y %H:%M:%S'
DATE_FILE_FORMAT = '%Y%m%d-%H%M%S'


if __name__ == "__main__":
    print("Hello")
    d = dt.datetime(2026, 2, 15, 10, 50, 7)
    print(d.strftime(DATE_FILE_FORMAT))
