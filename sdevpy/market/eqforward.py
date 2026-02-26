import os, json
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.tools import dates, timegrids
from sdevpy.maths import interpolation as itp
from sdevpy.market import yieldcurve as ycrv
from sdevpy.market.spot import get_spots


def get_forward_curves(names, valdate, **kwargs):
    spots = get_spots(names, valdate)

    fwd_curves = []
    for name, spot in zip(names, spots):
        file = data_file(name, valdate, **kwargs)
        data = eqforwarddata_from_file(file)
        curve = EqForwardCurve(valdate=valdate, interp_var='forward', interp_type='cubicspline')
        # yieldcurve = ycrv.get_yieldcurve('USD.SOFR.1D', valdate)
        curve.calibrate(data, spot)#, yieldcurve)
        fwd_curves.append(curve)

    # drifts = np.asarray([0.02, 0.05, 0.04])
    # fwd_curves_old = []
    # for s, mu in zip(spots, drifts):
    #     # Use the default variable trick to circumvent late binding in python loops
    #     # Otherwise, all the lambda functions will effectively be the same
    #     fwd_curves_old.append(lambda t, s=s, mu=mu: s * np.exp(mu * t))

    return fwd_curves


class EqForwardData:
    def __init__(self, valdate, pillars, **kwargs):
        self.valdate = valdate
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
            pillar = {'expiry': expiry_str, 'forward': forward}
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

    def calibrate(self, data, spot, yieldcurve=None):
        """ The data input is of type EqForwardData which contains pre-sorted data """
        self.spot = spot
        self.yieldcurve = yieldcurve
        self.dates = data.expiries
        self.fwds = data.forwards

        # Check consistency
        for d in self.dates:
            if d <= self.valdate:
                raise RuntimeError(f"Expiry before or at valuation date: {d.strftime(dates.DATE_FORMAT)}")

        # Calibrate
        if self.interp_var == 'yield':
            if yieldcurve is None:
                raise RuntimeError("Yield curve must be provided for dividend yield-based EQ forward curve construction")

            int_dates, calc_fwds = np.asarray(self.dates), np.asarray(self.fwds)
            int_times = timegrids.model_time(self.valdate, int_dates)
            # Calculate yields
            yields = np.log(spot * yieldcurve.discount(int_dates) / calc_fwds) / int_times
            self.interp = itp.create_interpolation(interp=self.interp_type, l_extrap='flat', r_extrap='flat')
            self.interp.set_data(int_times, yields)
        elif self.interp_var == 'forward':
            int_dates, int_fwds = [self.valdate], [self.spot]
            int_dates.extend(self.dates)
            int_fwds.extend(self.fwds)
            int_times = timegrids.model_time(self.valdate, int_dates)
            self.interp = itp.create_interpolation(interp=self.interp_type, l_extrap='flat', r_extrap='flat')
            self.interp.set_data(int_times, int_fwds)
        else:
            raise RuntimeError(f"Unknown forward interpolation variable: {self.interp_var}")

    def value(self, date):
        t = timegrids.model_time(self.valdate, date)
        fwd = self.value_float(t)
        return fwd

    def value_float(self, t):
        if self.interp_var == 'yield':
            if self.yieldcurve is None:
                raise RuntimeError("Missing yield curve when calculating EQ forward using dividend yields")

            y = self.interp.value(t)
            return self.spot * self.yieldcurve.discount_float(t) * np.exp(-y * t)
        elif self.interp_var == 'forward':
            return self.interp.value(t)
        else:
            raise RuntimeError(f"Unknown forward interpolation variable: {self.interp_var}")


def eqforwarddata_from_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    pillars = data.get('pillars')

    # Convert date strings into dates
    for pillar in pillars:
        date_str = pillar.get('expiry')
        date = dt.datetime.strptime(date_str, dates.DATE_FORMAT)
        pillar['expiry'] = date

    data = EqForwardData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), pillars,
                         name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT))
    return data


def data_file(name, date, **kwargs):
    folder = kwargs.get('folder', test_data_folder())
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
    import matplotlib.pyplot as plt

    name = "ABC"
    valdate = dt.datetime(2026, 2, 15)

    # Generate a sample to start from
    spot = 100.0
    expiries = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
                dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15)]
    zrs = np.asarray([-0.01, -0.015, -0.020, -0.022, -0.025])
    ztimes = timegrids.model_time(valdate, expiries)
    zfwds = spot * np.exp(-zrs * ztimes)

    pillars = [{'expiry': d, 'forward': f} for d, f in zip(expiries, zfwds)]
    data = EqForwardData(valdate, pillars, name=name)
    file = data_file(name, valdate)
    data.dump(file)

    # Get data from existing file
    test_data = eqforwarddata_from_file(file)

    # Create forward curve
    curve = EqForwardCurve(valdate=valdate, interp_var='forward', interp_type='cubicspline')
    yieldcurve = ycrv.get_yieldcurve('USD.SOFR.1D', valdate)
    curve.calibrate(test_data, spot, yieldcurve)

    # Interpolate and display
    test_dates = [dates.date_advance(valdate, months=1*n) for n in range(1, 150)]
    test_fwds = curve.value(test_dates)

    # Original data
    base_dates = test_data.expiries
    base_fwds = test_data.forwards

    plt.plot(test_dates, test_fwds)
    plt.scatter(base_dates, base_fwds, color='black')
    plt.show()
