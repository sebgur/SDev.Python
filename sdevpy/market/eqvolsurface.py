import os, json, logging
import datetime as dt
import numpy as np
import numpy.typing as npt
from pathlib import Path
from sdevpy.utilities import dates
from sdevpy.utilities import timegrids
from sdevpy.analytics import black
from sdevpy.market.eqforward import EqForwardCurve, get_forward_curves
log = logging.getLogger(Path(__file__).stem)


class EqVolSurfaceData:
    def __init__(self, valdate: dt.datetime, sections, **kwargs):
        self.valdate = valdate
        self.name = kwargs.get('name', '')
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.strike_input_type = kwargs.get('strike_input_type', 'absolute').lower()

        # Sort by increasing date
        sections.sort(key=lambda x: x['expiry'])

        # Extract
        self.expiries = np.asarray([s['expiry'] for s in sections])
        self.input_strikes = [np.asarray(s['strikes']) for s in sections]
        self.vols = [np.asarray(s['vols']) for s in sections]

        # Size checks
        n_times = len(self.expiries)
        if any(len(x) != n_times for x in (self.input_strikes, self.vols)):
            raise ValueError("Incompatible size along time direction between expiries, forwards, strikes and vols")

    def get_prices(self, fwd_curve: EqForwardCurve, option_type: str='call') -> npt.ArrayLike:
        option_type_lw = option_type.lower()
        prices = []
        abs_strikes = self.get_strikes(fwd_curve, to_type='absolute')
        for exp_idx, expiry in enumerate(self.expiries):
            t = timegrids.model_time(self.valdate, expiry)
            fwd = fwd_curve.value(expiry)
            strikes = abs_strikes[exp_idx]
            vols = self.vols[exp_idx]
            match option_type_lw:
                case 'call':
                    price = black.price(t, strikes, True, fwd, vols)
                case 'put':
                    price = black.price(t, strikes, False, fwd, vols)
                case 'straddle':
                    price = black.price(t, strikes, True, fwd, vols)
                    price = price + black.price(t, strikes, False, fwd, vols)
                case _:
                    raise ValueError(f"Invalid option type: {option_type}")

            prices.append(price)
        return prices

    def get_strikes(self, fwd_curve: EqForwardCurve=None, to_type: str='absolute') -> npt.ArrayLike:
        """ Retrieve strikes, absolute or relative """
        to_type_lw = to_type.lower()
        if to_type_lw == self.strike_input_type:
            return self.input_strikes
        else: # Need conversion
            if fwd_curve is None:
                raise ValueError(f"Forward curve required for strike conversion but None given: {self.name}")

            # Need to loop over expiries because not all expiries must have the same number of strikes.
            # Therefore we cannot put the strikes into numpy arrays.
            fwds = fwd_curve.value(self.expiries)
            n_times = len(self.expiries)
            if to_type_lw == 'absolute' and self.strike_input_type == 'relative':
                conv_strikes = [self.input_strikes[i] * fwds[i] for i in range(n_times)]
            elif to_type_lw == 'relative' and self.strike_input_type == 'absolute':
                conv_strikes = [self.input_strikes[i] / fwds[i] for i in range(n_times)]
            else:
                raise ValueError(f"Unknown strike type {to_type}: expected absolute or relative")

            return conv_strikes

    def dump(self, file, indent=2):
        data = self.dump_data()
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)

    def dump_data(self):
        sections = []
        for i, expiry in enumerate(self.expiries):
            expiry_str = expiry.strftime(dates.DATE_FORMAT)
            section = {'expiry': expiry_str, 'strikes': self.input_strikes[i].tolist(),
                       'vols': self.vols[i].tolist()}
            sections.append(section)

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'strike_input_type': self.strike_input_type, 'sections': sections}

        return data

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
            with np.printoptions(precision=n_digits):
                print("Strikes", self.input_strikes[i])
                print("Vols", self.vols[i])

        print(sep)
        print(sep)


def eqvolsurfacedata_from_file(file: str) -> EqVolSurfaceData:
    """ Retrieve EqVolSurfaceData from file """
    with open(file) as f:
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

    data = EqVolSurfaceData(dt.datetime.strptime(valdate, dates.DATE_FORMAT), sections,
                            name=name, snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT),
                            strike_input_type=strike_input_type)
    return data


def data_file(name: str, date: dt.datetime, **kwargs) -> str:
    """ Data file path for EQ vol surfaces """
    folder = kwargs.get('folder', test_data_folder())
    name_folder = os.path.join(folder, name)
    os.makedirs(name_folder, exist_ok=True)
    file = os.path.join(name_folder, date.strftime(dates.DATE_FILE_FORMAT) + ".json")
    return file


def test_data_folder() -> str:
    """ Test data folder for EQ vol surfaces """
    folder = Path(__file__).parent.parent.parent.resolve()
    dataset_folder = os.path.join(folder, "datasets")
    folder = os.path.join(os.path.join(dataset_folder, "marketdata"), "eqoptions")
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

    # Get forward curve
    fwd_curve = get_forward_curves([name], valdate)[0]

    # Get data from existing file
    file = data_file(name, valdate, folder=folder)
    surface_data = eqvolsurfacedata_from_file(file)
    print(surface_data.get_strikes(to_type='absolute'))
    print(surface_data.get_strikes(fwd_curve=fwd_curve, to_type='relative'))
    surface_data.pretty_print(4)
