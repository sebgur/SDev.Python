import os, json
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.tools import dates, timegrids
from sdevpy.maths import interpolation as itp


########### TODO ##################
# * Sounds like we had better split raw market data from the object.
# * Have an EqForwardData object that purely retrieves the forward data curve
# * Then have a modelling object which can evaluate

class EqForwardData:
    def __init__(self, valdate, pillars, **kwargs):
        self.valdate = kwargs.get('valdate', None)
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')

        # Sort by increasing date
        pillars.sort(key=lambda x: x['expiry'])

        # Extract
        self.expiries = np.asarray([s['expiry'] for s in pillars])
        self.forwards = np.asarray([s['forward'] for s in pillars])

    def dump(self, file, indent=2):
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)

    def dump_data(self):
        pillars = []
        for expiry, forward in zip(self.expiries, self.forwards):
            expiry_str = expiry.strftime(dates.DATE_FORMAT)
            pillar = {'expiry': expiry_str, 'forward': self.forwards[i]}
            pillars.append(pillar)

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT), 'pillars': pillars}
        return data


class EqForwardCurve:
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', None)
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')
        self.dates, self.fwds, self.spot, self.yieldcurve = None, None, None, None
        self.interp_var = kwargs.get('interp_var', 'yield').lower()
        self.interp_type = kwargs.get('interp_type', 'cubispline').lower()
        self.interp = None

    def calibrate(self, dates, fwds, spot, yieldcurve=None):
        self.spot = spot
        self.yieldcurve = yieldcurve

        # Check consistency
        for d in dates:
            if d <= self.valdate:
                raise RuntimeError(f"Expiry before or at valuation date: {d.strftime(dates.DATE_FORMAT)}")

        # Sort
        sorted_pillars = [{'expiry': d, 'forward': f} for d, f in zip(dates, fwds)]
        sorted_pillars.sort(lambda x: x['expiry'])
        self.dates = [p['expiry'] for p in sorted_pillars]
        self.fwds = [p['forward'] for p in sorted_pillars]

        # Calibrate
        if interp_var == 'yield':
            if yieldcurve is None:
                raise RuntimeError("Yield curve must be provided for dividend yield-based EQ forward curve construction")

            int_dates, calc_fwds = np.asarray(self.dates), np.asarray(self.fwds)
            int_times = timegrids.model_time(self.valdate, int_dates)
            # Calculate yields
            yields = np.log(spot * yieldcurve.discount(calc_dates) / calc_fwds) / int_times
            self.interp = itp.create_interpolation(interp=self.interp_type, l_extrap=flat, r_extrap='flat')
            self.interp.set_data(int_times, yields)
        elif interp_var == 'forward':
            int_dates, int_fwds = [self.valdate], [spot]
            int_dates.extend(self.dates)
            int_fwds.extend(self.fwds)
            int_times = timegrids.model_time(self.valdate, int_dates)
            self.interp = itp.create_interpolation(interp=self.interp_type, l_extrap=None, r_extrap='flat')
            self.interp.set_data(int_times, int_fwds)
        else:
            raise RuntimeError(f"Unknown forward interpolation variable: {interp_var}")

    def value(self, date):
        t = timegrids.model_time(self.valdate, date)
        if interp_var == 'yield':
            if self.yieldcurve is None:
                raise RuntimeError("Missing yield curve when calculating EQ forward using dividend yields")

            y = self.interp.value(t)
            return spot * self.yieldcurve.discount(date) * np.exp(-y * t)
        elif interp_var == 'forward':
            return self.interp.value(t)
        else:
            raise RuntimeError(f"Unknown forward interpolation variable: {interp_var}")


def eqforwarddata_from_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    pillars = data.get('pillars')

    # Convert date strings into dates
    for pillar in pillars:
        date_str = section.get('expiry')
        date = dt.datetime.strptime(date_str, dates.DATE_FORMAT)
        pillars['expiry'] = date

    data = EqForwardData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), pillars,
                         name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT))
    return data


def data_file(folder, name, date):
    name_folder = os.path.join(folder, name)
    os.makedirs(name_folder, exist_ok=True)
    file = os.path.join(name_folder, date.strftime(dates.DATE_FILE_FORMAT) + ".json")
    return file


def test_data_folder():
    folder = Path(__file__).parent.parent.parent.resolve()
    dataset_folder = os.path.join(folder, "datasets")
    folder = os.path.join(os.path.join(dataset_folder, "marketdata"), "eqforwards")
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    name = "ABC"
    valdate = dt.datetime(2025, 12, 15)
    folder = test_data_folder()

    # # Generate a sample to start from
    # from sdevpy.models import svivol
    # terms = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    # expiries, fwds, strike_surface, vol_surface = svivol.generate_sample_data(valdate, terms)
    # strike_types = ['absolute' for expiry in expiries]
    # surface_data = VolSufaceData(valdate, expiries, fwds, strike_surface, vol_surface, strike_types,
    #                              name=name)
    # file = data_file(folder, name, valdate)
    # surface_data.dump(file)

    # Get data from existing file
    file = data_file(folder, name, valdate)
    surface_data = vol_surface(file)
    print(surface_data.get_strikes('absolute'))
    print(surface_data.get_strikes('relative'))
    surface_data.pretty_print(4)
