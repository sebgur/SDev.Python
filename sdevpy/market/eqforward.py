import json
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.maths import interpolation as itp


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
        for expiry, forward in zip(self.expiries, self.forwards, strict=True):
            expiry_str = expiry.strftime(dts.DATE_FORMAT)
            pillar = {'expiry': expiry_str, 'forward': forward}
            pillars.append(pillar)

        data = {'name': self.name, 'valdate': self.valdate.strftime(dts.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dts.DATETIME_FORMAT), 'pillars': pillars}
        return data


class EqForwardCurve:
    def __init__(self, **kwargs):
        self.valdate = kwargs.get('valdate', None)
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')
        self.dates, self.fwds, self.spot, self.yieldcurve = None, None, None, None
        self.interp_var = kwargs.get('interp_var', 'yield').lower()
        self.interp_type = kwargs.get('interp_type', 'cubicspline').lower()
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
                raise RuntimeError(f"Expiry before or at valuation date: {d.strftime(dts.DATE_FORMAT)}")

        # Calibrate
        if self.interp_var == 'yield':
            if yieldcurve is None:
                raise RuntimeError("Yield curve must be provided for dividend yield-based forward curve construction")

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

    def value(self, date: dt.datetime|list[dt.datetime]):
        t = timegrids.model_time(self.valdate, date)
        fwd = self.value_float(t)
        return fwd

    def value_float(self, t: float):
        if self.interp_var == 'yield':
            if self.yieldcurve is None:
                raise RuntimeError("Missing yield curve when calculating EQ forward using dividend yields")

            y = self.interp.value(t)
            return self.spot * self.yieldcurve.discount_float(t) * np.exp(-y * t)
        elif self.interp_var == 'forward':
            return self.interp.value(t)
        else:
            raise RuntimeError(f"Unknown forward interpolation variable: {self.interp_var}")


def eqforwarddata_from_file(file: str|Path) -> EqForwardData:
    """ Retrieve EQ forward data from file """
    with open(file) as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    pillars = data.get('pillars')

    # Convert date strings into dates
    for pillar in pillars:
        date_str = pillar.get('expiry')
        date = dt.datetime.strptime(date_str, dts.DATE_FORMAT)
        pillar['expiry'] = date

    data = EqForwardData(dt.datetime.strptime(valdate, dts.DATE_FORMAT), pillars,
                         name=name, snapdate=dt.datetime.strptime(snapdate, dts.DATETIME_FORMAT))
    return data


if __name__ == "__main__":
    print("Hello")
