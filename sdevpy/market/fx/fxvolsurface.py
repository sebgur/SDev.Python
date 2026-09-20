import datetime as dt
from pathlib import Path
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import dates as dts
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.market.fx.fxforward import fx_pillar_date, fx_spot_date


class FxVolSurfaceData:
    def __init__(self, valdate: dt.datetime, sections: list[dict], **kwargs):
        self.valdate = valdate
        self.snapdate = kwargs.get('snapdate', self.valdate)
        self.name = kwargs.get('name', '')
        self.market_strangle_quote = kwargs.get('market_strangle_quote', False)
        self.spot_delta_cutoff = kwargs.get('spot_delta_cutoff', '1Y')

        sections.sort(key=lambda x: dts.tenor_to_date(x['tenor']))

        for idx, s in enumerate(sections):
            if not (len(s['deltas']) == len(s['rr']) == len(s['bf'])):
                raise ValueError(f"Mismatch in sizes between deltas, rr and bf on section index {idx}")

        # self.expiries = np.asarray([s['expiry'] for s in sections])
        self.tenors = np.asarray([s['tenor'] for s in sections])
        self.atm_vols = np.asarray([s['atm_vol'] for s in sections], dtype=float)
        self.deltas = [np.asarray(s['deltas'], dtype=float) for s in sections]
        self.rr = [np.asarray(s['rr'], dtype=float) for s in sections]
        self.bf = [np.asarray(s['bf'], dtype=float) for s in sections]

        if len(self.tenors) != len(sections):
            raise ValueError("Mismatch in sizes between tenors and sections")

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
        for i, tenor in enumerate(self.tenors):
            sections.append({'tenor': tenor,
                             'atm_vol': float(self.atm_vols[i]),
                             'deltas': self.deltas[i].tolist(),
                             'rr': self.rr[i].tolist(),
                             'bf': self.bf[i].tolist()})

        return {'name': self.name, 'valdate': self.valdate.strftime(dts.DATE_FORMAT),
                'snapdate': self.snapdate.strftime(dts.DATETIME_FORMAT),
                'market_strangle_quote': self.market_strangle_quote,
                'spot_delta_cutoff': self.spot_delta_cutoff, 'sections': sections}

    def pretty_print(self, n_digits: int = 4) -> None:
        """ Print information """
        sep = '-' * 70
        print(sep)
        print(sep)
        print(f"Name: {self.name}")
        print(f"Valuation date: {self.valdate.strftime(dts.DATE_FORMAT)}")
        print(f"Snap date: {self.snapdate.strftime(dts.DATETIME_FORMAT)}")
        print(f"Market strangle quote: {self.market_strangle_quote}")
        n_exp = len(self.tenors)
        print(f"Number of expiries: {n_exp}")
        for i in range(n_exp):
            print(sep)
            print(f"Expiry {i + 1}/{n_exp}: {self.tenors[i]}")
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
    spot_delta_cutoff = data.get('spot_delta_cutoff', '1Y')
    sections = data.get('sections')

    return FxVolSurfaceData(dt.datetime.strptime(valdate, dts.DATE_FORMAT), sections,
                            name=name, snapdate=dt.datetime.strptime(snapdate, dts.DATETIME_FORMAT),
                            market_strangle_quote=market_strangle_quote, spot_delta_cutoff=spot_delta_cutoff)


def wingvols_from_butterfly(atm_vol: float, rr: float, bf: float) -> float:
    """ Wing vols at the quoted delta, given atm_vol/rr/bf where bf is the butterfly """
    vol_call = atm_vol + bf + 0.5 * rr
    vol_put = atm_vol + bf - 0.5 * rr
    return vol_put, vol_call


def fx_option_dates(valdate: dt.datetime, tenor: str, forccy: str, domccy: str) -> tuple[dt.datetime, dt.datetime]:
    """ Calculate FX option expiry and delivery dates """
    expiry = fx_pillar_date(valdate, tenor, forccy, domccy)
    delivery = fx_spot_date(expiry, forccy, domccy)
    return expiry, delivery


if __name__ == "__main__":
    print("Hello")
