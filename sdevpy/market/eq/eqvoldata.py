import logging
import datetime as dt
import numpy as np
from sdevpy.utilities import dates
from sdevpy.utilities import jsonmanager as jsm
log = logging.getLogger(__name__)


class EqVolData:
    def __init__(self, valdate: dt.datetime, sections, **kwargs):
        self.valdate = valdate
        self.name = kwargs.get('name', '')
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.strike_input_type = kwargs.get('strike_input_type', 'absolute').lower()

        # Sort by increasing date
        sections.sort(key=lambda x: x['expiry'])

        # Size checks
        for idx in range(len(sections)):
            s = sections[idx]
            if len(s['strikes']) != len(s['vols']):
                raise ValueError(f"Mismatch in sizes between strikes and vols on section index {idx}")

        # Extract
        self.expiries = np.asarray([s['expiry'] for s in sections])
        self.input_strikes = [np.asarray(s['strikes']) for s in sections]
        self.vols = [np.asarray(s['vols']) for s in sections]

        # Size checks
        if len(self.expiries) != len(sections):
            raise ValueError("Mismatch in sizes between expiries and sections")

    def dump(self, file: str, indent: int=2) -> None:
        """ Dump data to file """
        data = self.dump_data()
        jsm.serialize(data, file, indent=indent)

    def dump_data(self) -> dict:
        """ Dump data as dictionary """
        sections = []
        for i, expiry in enumerate(self.expiries):
            expiry_str = expiry.strftime(dates.DATE_FORMAT)
            section = {'expiry': expiry_str, 'strikes': self.input_strikes[i].tolist(),
                       'vols': self.vols[i].tolist()}
            sections.append(section)

        data = {'name': self.name, 'valdate': self.valdate.strftime(dates.DATETIME_FORMAT),
                'snapdate': self.snapdate.strftime(dates.DATETIME_FORMAT),
                'strike_input_type': self.strike_input_type, 'sections': sections}

        return data

    def pretty_print(self, n_digits: int=4) -> None:
        """ Print information """
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


def eqvoldata_from_file(file: str) -> EqVolData:
    """ Retrieve EqVolData from file """
    data = jsm.deserialize(file)

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

    data = EqVolData(dt.datetime.strptime(valdate, dates.DATETIME_FORMAT), sections, name=name,
                     snapdate=dt.datetime.strptime(snapdate, dates.DATETIME_FORMAT),
                     strike_input_type=strike_input_type)
    return data


if __name__ == "__main__":
    print("Hello")
