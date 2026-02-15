import os, json
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.tools import dates


class VolSufaceData:
    def __init__(self, valdate, expiries, fwds, strikes, vols, strike_types, **kwargs):
        self.valdate = valdate
        self.expiries = expiries
        self.forwards = fwds
        self.strikes = strikes
        self.vols = vols
        self.strike_types = strike_types
        self.name = kwargs.get('name', 'MyIndex')
        self.snapdate = kwargs.get('snapdate', valdate)

    def dump(self, file):
        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT)}

        sections = []
        for i, expiry in enumerate(self.expiries):
            expiry_str = expiry.strftime(dates.DATE_FORMAT)
            section = {'expiry': expiry_str, 'forward': self.forwards[i], 'strikes': self.strikes[i].tolist(),
                       'vols': self.vols[i].tolist(), 'strike_type': self.strike_types[i]}
            sections.append(section)

        data['sections'] = sections

        with open(file, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def read(file):
        with open(file, 'r') as f:
            data = json.load(f)

        name = data.get('name')
        valdate = data.get('valdate')
        snapdate = data.get('snapdate')
        sections = data.get('sections')
        # Convert date strings into dates
        for section in sections:
            date_str = section.get('expiry')
            date = dt.datetime.strptime(date_str, dates.DATE_FORMAT)
            section['expiry'] = date

        # Sort by increasing date
        sections.sort(key=lambda x: x['expiry'])

        # Extract
        expiries = np.asarray([s['expiry'] for s in sections])
        forwards = np.asarray([s['forward'] for s in sections])
        strikes = np.asarray([s['strikes'] for s in sections])
        vols = np.asarray([s['vols'] for s in sections])
        strike_types = np.asarray([s['strike_type'] for s in sections])

        voldata = VolSufaceData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), expiries, forwards, strikes, vols, strike_types,
                                name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT))
        return voldata


def data_file(folder, name, date, extension='json'):
    name_folder = os.path.join(folder, name)
    os.makedirs(name_folder, exist_ok=True)
    file = os.path.join(name_folder, date.strftime(dates.DATE_FILE_FORMAT) + "." + extension)
    return file


if __name__ == "__main__":
    name = "MyIndex"
    valdate = dt.datetime(2025, 12, 15)
    folder = Path(__file__).parent.parent.parent.resolve()
    folder = os.path.join(os.path.join(folder, "datasets"), "impliedvols")
    os.makedirs(folder, exist_ok=True)

    # Generate for first sample
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
    surface_data = VolSufaceData.read(file)
    print(surface_data)
