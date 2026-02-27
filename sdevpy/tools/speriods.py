from dateutil.relativedelta import relativedelta
import re


def period(tenor_str) -> relativedelta:
    pattern = r'(\d+Y)?(\d+M)?(\d+W)?(\d+D)?'
    match = re.fullmatch(pattern, tenor_str.upper())
    if not match or not any(match.groups()):
        raise ValueError(f"Invalid tenor string: '{tenor_str}'")

    years   = int(match.group(1)[:-1]) if match.group(1) else 0
    months  = int(match.group(2)[:-1]) if match.group(2) else 0
    weeks   = int(match.group(3)[:-1]) if match.group(3) else 0
    days    = int(match.group(4)[:-1]) if match.group(4) else 0

    return relativedelta(years=years, months=months, weeks=weeks, days=days)


if __name__ == "__main__":
    import datetime as dt
    base = dt.datetime(2025, 12, 15)
    print(f"Base: {base}")
    tenors = ['1D', '2W', '1M', '2Y', '1Y6M']
    for t in tenors:
        print(f"{t}- {base + period(t)}")
