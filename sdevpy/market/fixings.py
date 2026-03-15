import os
import datetime as dt
import pandas as pd
from pathlib import Path
from sdevpy.tools import dates as dts
from sdevpy.maths import interpolation as itp


def get_fixings(names, dates, **kwargs):
    fixings = None
    return fixings


class FixingHandler:
    def __init__(self, name, dates, values, **kwargs):
        self.name = name
        self.interpolate = kwargs.get('interpolate', False)

        # Sort by increasing date
        data = [{'date': d, 'value': v} for d, v in zip(dates, values, strict=True)]
        data.sort(key=lambda x: x['date'])

        # Extract
        self.dates = [s['date'] for s in data]
        self.values = [s['value'] for s in data]

        # Set interpolation if needed
        if self.interpolate:
            self.oadates = dts.to_oadate(self.dates)
            self.interp = itp.create_interpolation(l_extrap='none', r_extrap='none')
            self.interp.set_data(self.oadates, self.values)
        else:
            self.oadates, self.interp = None, None

    def values(self, dates):
        return [self.value(d) for d in dates]

    def value(self, date):
        try:
            idx = self.dates.index(date)
            return self.values[idx]
        except Exception as e:
            if self.interpolate:
                t = dts.to_oadate(date)
                fixing = self.interpolate_values(t)
                return fixing
            else:
                msg = f"No fixing found for {self.name} and date {date.strftime(dts.DATETIME_FORMAT)}"
                raise RuntimeError(msg) from e

    def interpolate_values(self, t):
        if self.interp is None:
            raise RuntimeError("Fixing interpolation not turned on")

        return self.interp.value(t)


def fixinghandler(name, interpolate=False, **kwargs):
    file = data_file(name, **kwargs)
    df = pd.read_csv(file, parse_dates=["Date"], dtype={"Value": float})

    # Create data object
    dates = df['Date'].tolist()
    values = df['Value'].tolist()
    data = FixingHandler(name, dates, values, interpolate=interpolate)
    return data


def add_fixing(name, dates, values, **kwargs):
    file = data_file(name, **kwargs)

    # Format new data
    new_df = pd.DataFrame({"Date": dates, "Value": values})
    new_df['Date'] = new_df['Date'].dt.strftime(dts.DATE_FORMAT)

    # Append
    new_df.to_csv(file, mode='a', header=False, index=False)

    # Check
    if kwargs.get('check', False):
        check_fixings(name)


def check_fixings(name, **kwargs):
    """ Diagnostic tool reading existing series, ordering it and checking for duplicates """
    # Retrieve existing data
    file = data_file(name, **kwargs)
    df = pd.read_csv(file)

    # Parse to dates
    df["Date"] = pd.to_datetime(df["Date"], format=dts.DATE_FORMAT)

    # Check for duplicated
    dupes = df[df['Date'].duplicated(keep=False)]
    if len(dupes) > 0:
        dupe_file = file.replace(".csv", ".dupes.csv")
        dupes.to_csv(dupe_file, index=False)
        raise ValueError(f"Duplicates found and output to {dupe_file}. Remove before continuing.")

    # Sort
    df = df.sort_values("Date")
    df["Date"] = df["Date"].dt.strftime(dts.DATE_FORMAT)

    # Reoutput
    df.to_csv(file, index=False)


def data_file(name, **kwargs):
    folder = kwargs.get('folder', test_data_folder())
    file = os.path.join(folder, name + ".csv")
    return file


def test_data_folder():
    folder = Path(__file__).parent.parent.parent.resolve()
    dataset_folder = os.path.join(folder, "datasets")
    folder = os.path.join(os.path.join(dataset_folder, "marketdata"), "fixings")
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    name = "ABC"

    # # Diagnostics
    # check_fixings(name)

    # # Add some entries
    # new_dates = [dt.datetime(2024, 2, 10), dt.datetime(2023, 1, 5)]
    # new_values = [10.0, 20.0]
    # add_fixing(name, new_dates, new_values)

    # Retrieve data
    data = fixinghandler(name, interpolate=True)
    # f = data.value(dt.datetime(2025, 1, 2))
    f = data.value(dt.datetime(2025, 9, 15))
    print(f)

