import os
import datetime as dt
import pandas as pd
# import numpy as np
from pathlib import Path
# from sdevpy.tools import dates#, timegrids
from sdevpy.maths import interpolation as itp


def get_fixings(names, dates, **kwargs):
    fixings = None
    return fixings


class FixingData:
    def __init__(self, name, dates, values, **kwargs):
        self.name = name
        self.interpolate = kwargs.get('interpolate', False)

        # Sort by increasing date
        data = {'date': dates, 'value': values}
        data.sort(key=lambda x: x['date'])

        # Extract
        self.dates = [s['date'] for s in data]
        self.values = [s['value'] for s in data]

        # Set interpolation if needed
        if self.interpolate:
            self.oadates = dates.to_oadate(self.dates)
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
                t = dates.to_oadate(date)
                fixing = self.value_float(t)
                return fixing
            else:
                msg = f"No fixing found for {self.name} and date {date.strftime(dates.DATETIME_FORMAT)}"
                raise RuntimeError(msg) from e

    def value_float(self, t):
        if self.interp is None:
            raise RuntimeError("Fixing interpolation not turned on")

        return self.interp.value(t)


def fixingdata(name, **kwargs):
    file = data_file(name, **kwargs)
    df = pd.read_csv(file, parse_dates=["Date"], dtype={"Value": float})

    # Create data object
    dates = df['Date']
    values = df['Value']
    data = FixingData(name, dates, values)
    return data


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

    # Retrieve data
    data = fixingdata(name)
    print(data)

    # # Generate a sample to start from
    # spot = 100.0
    # expiries = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
    #             dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15)]
    # zrs = np.asarray([-0.01, -0.015, -0.020, -0.022, -0.025])
    # ztimes = timegrids.model_time(valdate, expiries)
    # zfwds = spot * np.exp(-zrs * ztimes)

    # pillars = [{'expiry': d, 'forward': f} for d, f in zip(expiries, zfwds)]
    # data = EqForwardData(valdate, pillars, name=name)
    # file = data_file(name, valdate)
    # data.dump(file)

    # # Get data from existing file
    # test_data = eqforwarddata_from_file(file)

    # # Create forward curve
    # curve = EqForwardCurve(valdate=valdate, interp_var='forward', interp_type='cubicspline')
    # yieldcurve = ycrv.get_yieldcurve('USD.SOFR.1D', valdate)
    # curve.calibrate(test_data, spot, yieldcurve)

    # # Interpolate and display
    # test_dates = [dates.advance(valdate, months=1*n) for n in range(1, 150)]
    # test_fwds = curve.value(test_dates)

    # # Original data
    # base_dates = test_data.expiries
    # base_fwds = test_data.forwards
