import os, json
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.tools import dates


class VolSufaceData:
    def __init__(self, valdate, sections, **kwargs):
        self.valdate = valdate
        self.name = kwargs.get('name', 'MyIndex')
        self.snapdate = kwargs.get('snapdate', valdate)
        self.strike_input_type = kwargs.get('strike_input_type', 'absolute').lower()

        # Sort by increasing date
        sections.sort(key=lambda x: x['expiry'])

        # Extract
        self.expiries = np.asarray([s['expiry'] for s in sections])
        self.forwards = np.asarray([s['forward'] for s in sections])
        self.input_strikes = np.asarray([s['strikes'] for s in sections])
        self.vols = np.asarray([s['vols'] for s in sections])

        # self.expiries = np.asarray(expiries)
        # self.forwards = np.asarray(fwds)
        # self.vols = np.asarray(vols)
        # self.input_strikes = np.asarray(strikes)

        # Calculate missing strike type
        if self.strike_input_type == 'absolute':
            self.abs_strikes = self.input_strikes
        elif self.strike_input_type == 'relative':
            self.abs_strikes = self.forwards * self.input_strikes
        else:
            raise RuntimeError(f"Strike input type not supported yet: {self.strike_input_type}")

    def get_strikes(self, type='absolute'):
        req_type = type.lower()
        if req_type == 'absolute':
            return self.abs_strikes
        elif req_type == 'relative':
            return self.abs_strikes / self.forwards.reshape(-1, 1)

    def dump(self, file):
        sections = []
        for i, expiry in enumerate(self.expiries):
            expiry_str = expiry.strftime(dates.DATE_FORMAT)
            section = {'expiry': expiry_str, 'forward': self.forwards[i], 'strikes': self.input_strikes[i].tolist(),
                       'vols': self.vols[i].tolist()}
            sections.append(section)

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'strike_input_type': self.strike_input_type, 'sections': sections}

        with open(file, 'w') as f:
            json.dump(data, f, indent=2)

    def pretty_print(self, n_digits=4):
        sep = '-'*70
        print(sep)
        print(sep)
        print(f"Name: {self.name}")
        print(f"Valuation date: {self.valdate.strftime(dates.DATE_FORMAT)}")
        print(f"Snap date: {self.snapdate.strftime(dates.DATETIME_FORMAT)}")
        print(f"Strike input type: {self.strike_input_type}")
        n_exp = len(self.expiries)
        print(f"Number of expiries: {n_exp}")
        for i in range(n_exp):
            print(sep)
            print(f"Expiry {i+1}/{n_exp}: {self.expiries[i].strftime(dates.DATE_FORMAT)}")
            print(f"Forward: {self.forwards[i]:,.{n_digits}f}")
            with np.printoptions(precision=n_digits):
                print("Strikes", self.input_strikes[i])
                print("Vols", self.vols[i])

        print(sep)
        print(sep)


def vol_surface(file):
    with open(file, 'r') as f:
        data = json.load(f)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    strike_input_type = data.get('strike_input_type')
    sections = data.get('sections')

    # Convert date strings into dates
    for section in sections:
        date_str = section.get('expiry')
        date = dt.datetime.strptime(date_str, dates.DATE_FORMAT)
        section['expiry'] = date

    # # Sort by increasing date
    # sections.sort(key=lambda x: x['expiry'])

    # # Extract
    # expiries = np.asarray([s['expiry'] for s in sections])
    # forwards = np.asarray([s['forward'] for s in sections])
    # strikes = np.asarray([s['strikes'] for s in sections])
    # vols = np.asarray([s['vols'] for s in sections])

    # voldata = VolSufaceData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), expiries, forwards, strikes, vols,
    #                         name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT),
    #                         strike_input_type=strike_input_type)

    voldata = VolSufaceData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), sections,
                            name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT),
                            strike_input_type=strike_input_type)

    return voldata


def data_file(folder, name, date, extension='json'):
    name_folder = os.path.join(folder, name)
    os.makedirs(name_folder, exist_ok=True)
    file = os.path.join(name_folder, date.strftime(dates.DATE_FILE_FORMAT) + "." + extension)
    return file


def test_data_file(name, date, extension='json'):
    folder = test_data_folder()
    return data_file(folder, name, date, extension=extension)


def test_data_folder():
    folder = Path(__file__).parent.parent.parent.resolve()
    folder = os.path.join(os.path.join(folder, "datasets"), "impliedvols")
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    name = "MyIndex"
    valdate = dt.datetime(2025, 12, 15)
    folder = test_data_folder()
    # folder = Path(__file__).parent.parent.parent.resolve()
    # folder = os.path.join(os.path.join(folder, "datasets"), "impliedvols")
    # os.makedirs(folder, exist_ok=True)

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
    surface_data = vol_surface(file)
    print(surface_data.get_strikes('absolute'))
    print(surface_data.get_strikes('relative'))

    surface_data.pretty_print(4)
