import os
import datetime as dt
import pandas as pd
from pathlib import Path
from sdevpy.tools import dates as dts
from sdevpy.maths import interpolation as itp


###### TODO #################
# Generate real-looking fixing data
# Need to change the name of method values() in FixingHandler as
# it clashes with its self.values member data.
# Or better: vectorize it


def get_fixings(name: str, dates, **kwargs):
    interpolate = kwargs.get('interpolate', False)
    handler = fixinghandler(name, interpolate=interpolate, **kwargs)
    return handler.values(dates)


class FixingHandler:
    def __init__(self, name: str, dates, values, **kwargs):
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
        except ValueError as e:
            if self.interpolate:
                t = dts.to_oadate(date)
                fixing = self.interpolate_values(t)
                return fixing
            else:
                msg = f"No fixing found for {self.name} and date {date.strftime(dts.DATETIME_FORMAT)}"
                raise ValueError(msg) from e

    def interpolate_values(self, t):
        if self.interp is None:
            raise ValueError("Fixing interpolation not turned on")

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


def check_fixings(name: str, **kwargs) -> None:
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


def data_file(name: str, **kwargs) -> str:
    """ Returns file path corresponding to name
        Args:
            **kwargs: folder (str, optional): defaults to test data folder
    """
    folder = kwargs.get('folder', test_data_folder())
    file = os.path.join(folder, name + ".csv")
    return file


def test_data_folder():
    """ Returns test data folder for fixings """
    folder = Path(__file__).parent.parent.parent.resolve()
    dataset_folder = os.path.join(folder, "datasets")
    folder = os.path.join(os.path.join(dataset_folder, "marketdata"), "fixings")
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    import numpy as np
    from sdevpy.tools.scalendar import make_schedule
    from sdevpy.tools.timegrids import model_time
    from sdevpy.maths.rand.rng import get_rng
    name = "ABC"

    # # Diagnostics
    # check_fixings(name)

    # # Add some entries
    # new_dates = [dt.datetime(2024, 2, 10), dt.datetime(2023, 1, 5)]
    # new_values = [10.0, 20.0]
    # add_fixing(name, new_dates, new_values)

    # # Retrieve data
    # data = fixinghandler(name, interpolate=True)
    # f = data.value(dt.datetime(2025, 9, 15))
    # print(f)

    #### Generate simulated data ####
    valdate = dt.datetime(2025, 12, 15)
    start_date = dt.datetime(2025, 11, 15)
    expiry = dt.datetime(2026, 12, 15)
    all_dates = make_schedule("USD", start_date, expiry, '1D')
    sim_dates = [date for date in all_dates if date < valdate]

    # Quick simulation to generate sample
    spot_s = 100
    vol = 0.25
    sim_values = [spot_s]
    rng = get_rng()
    gaussians = rng.normal(len(sim_dates) - 1)
    for i in range(len(sim_dates) - 1):
        ds = sim_dates[i]
        de = sim_dates[i + 1]
        dt = model_time(ds, de)
        stdev = vol * np.sqrt(dt)
        g = gaussians[i][0]
        spot_e = spot_s * np.exp(-0.5 * stdev * stdev + stdev * g)
        sim_values.append(spot_e)
        spot_s = spot_e

    # Output sample to file
    sim_values = [round(v, 4) for v in sim_values]
    print(sim_values)
    add_fixing(name, sim_dates, sim_values)
