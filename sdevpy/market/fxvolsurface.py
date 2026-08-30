"""
`market_strangle_quote` says whether `bf` is the broker's market strangle or a smile butterfly. It applies to
every section, since it's a property of the data source's convention, not of any one tenor.
"""
import datetime as dt
from pathlib import Path
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import dates as dts
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.market import fxspot
from sdevpy.tests import conftest


class FxVolSurfaceData:
    def __init__(self, valdate: dt.datetime, sections: list[dict], **kwargs):
        self.valdate = valdate
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')
        self.market_strangle_quote = kwargs.get('market_strangle_quote', False)

        sections.sort(key=lambda x: x['expiry'])

        for idx, s in enumerate(sections):
            if not (len(s['deltas']) == len(s['rr']) == len(s['bf'])):
                raise ValueError(f"Mismatch in sizes between deltas, rr and bf on section index {idx}")

        self.expiries = np.asarray([s['expiry'] for s in sections])
        self.atm_vols = np.asarray([s['atm_vol'] for s in sections], dtype=float)
        self.deltas = [np.asarray(s['deltas'], dtype=float) for s in sections]
        self.rr = [np.asarray(s['rr'], dtype=float) for s in sections]
        self.bf = [np.asarray(s['bf'], dtype=float) for s in sections]

        if len(self.expiries) != len(sections):
            raise ValueError("Mismatch in sizes between expiries and sections")

    def wings_at(self, expiry_idx: int) -> tuple[float, npt.NDArray, npt.NDArray, npt.NDArray]:
        """ (atm_vol, deltas, rr, bf) for the given expiry index """
        return (self.atm_vols[expiry_idx], self.deltas[expiry_idx],
                self.rr[expiry_idx], self.bf[expiry_idx])

    def dump(self, file: str | Path, indent: int = 2) -> None:
        """ Dump data to file """
        jsm.serialize(self.dump_data(), file, indent=indent)

    def dump_data(self) -> dict:
        """ Dump data as dictionary """
        sections = []
        for i, expiry in enumerate(self.expiries):
            sections.append({'expiry': expiry.strftime(dts.DATE_FORMAT),
                             'atm_vol': float(self.atm_vols[i]),
                             'deltas': self.deltas[i].tolist(),
                             'rr': self.rr[i].tolist(),
                             'bf': self.bf[i].tolist()})

        return {'name': self.name, 'valdate': self.valdate.strftime(dts.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dts.DATETIME_FORMAT),
                'market_strangle_quote': self.market_strangle_quote, 'sections': sections}

    def pretty_print(self, n_digits: int = 4) -> None:
        """ Print information """
        sep = '-' * 70
        print(sep)
        print(sep)
        print(f"Name: {self.name}")
        print(f"Valuation date: {self.valdate.strftime(dts.DATE_FORMAT)}")
        print(f"Snap date: {self.snapdate.strftime(dts.DATETIME_FORMAT)}")
        print(f"Market strangle quote: {self.market_strangle_quote}")
        n_exp = len(self.expiries)
        print(f"Number of expiries: {n_exp}")
        for i in range(n_exp):
            print(sep)
            print(f"Expiry {i + 1}/{n_exp}: {self.expiries[i].strftime(dts.DATE_FORMAT)}")
            with np.printoptions(precision=n_digits):
                print(f"ATM vol: {self.atm_vols[i]:.{n_digits}f}")
                print("Deltas  ", self.deltas[i])
                print("RR      ", self.rr[i])
                print("BF      ", self.bf[i])

        print(sep)
        print(sep)


def fxvolsurfacedata_from_file(file: str|Path) -> FxVolSurfaceData:
    """ Retrieve FxVolSurfaceData from file """
    data = jsm.deserialize(file)

    name = data.get('name')
    valdate = data.get('valdate')
    snapdate = data.get('snapdate')
    market_strangle_quote = data.get('market_strangle_quote', False)
    sections = data.get('sections')

    for section in sections:
        date_str = section.get('expiry')
        section['expiry'] = dt.datetime.strptime(date_str, dts.DATE_FORMAT)

    return FxVolSurfaceData(dt.datetime.strptime(valdate, dts.DATE_FORMAT), sections,
                            name=name, snapdate=dt.datetime.strptime(snapdate, dts.DATETIME_FORMAT),
                            market_strangle_quote=market_strangle_quote)


if __name__ == "__main__":
    print("Hello")
