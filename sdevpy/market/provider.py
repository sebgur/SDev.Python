import datetime as dt
import numpy as np
from pathlib import Path
from typing import Protocol, runtime_checkable
from sdevpy.utilities import dates as dts
from sdevpy.market import spot as spot_mod
from sdevpy.market import correlations
from sdevpy.market import yieldcurve as ycrv
from sdevpy.market.yieldcurve import YieldCurve
from sdevpy.market import eqforward as eqfwd
from sdevpy.market import eqvolsurface as eqvol
from sdevpy.market import fixings
from sdevpy.market.spot import SpotData
from sdevpy.market.eqforward import EqForwardData, EqForwardCurve
from sdevpy.market.eqvolsurface import EqVolSurfaceData
from sdevpy.market.fixings import FixingHandler
from sdevpy.tests import testconfig


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve: ...
    def get_fixings(self, name: str, dates: dt.datetime|list[dt.datetime], **kwargs) -> list[float]: ...
    def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler: ...
    def get_correlations(self, names: list[str], date: dt.datetime) -> np.ndarray: ...
    def get_spot(self, name: str, date: dt.datetime) -> float: ...
    def get_spots(self, names: list[str], date: dt.datetime) -> np.ndarray: ...
    def get_spot_data(self, name: str, date: dt.datetime) -> SpotData: ...

    def get_eq_forward_data(self, name: str, date: dt.datetime) -> EqForwardData: ...
    def get_eq_vol_data(self, name: str, date: dt.datetime) -> EqVolSurfaceData: ...


class MarketDataFileProvider:
    """ Reads market data from files on disk """
    def __init__(self, root: str|Path=None):
        self.root = (Path(root) if root is not None else testconfig.marketdata_path())

    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve:
        """ Retrieve yield curve """
        folder = self.root / 'yieldcurves'
        file = Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + ".json")
        curve = ycrv.yieldcurve_from_file(file)
        return curve

    def get_fixings(self, name: str, dates: dt.datetime|list[dt.datetime], interpolate: bool=False) -> list[float]:
        """ Retrieve fixings """
        handler = self.get_fixing_handler(name, interpolate=interpolate)
        return handler.value(dates)

    def get_fixing_handler(self, name: str, interpolate: bool=False) -> FixingHandler:
        """ Retrieve fixings handler """
        folder = self.root / 'fixings'
        return fixings.fixinghandler(name, interpolate=interpolate, folder=folder)

    def get_correlations(self, names: list[str], date: dt.datetime) -> np.ndarray:
        """ Retrieve correlations """
        folder = self.root / 'correlations'
        return correlations.get_correlations(names, date, folder=folder)

    def get_spot(self, name: str, date: dt.datetime) -> float:
        """ Retrieve spot """
        return self.get_spot_data(name, date).value

    def get_spots(self, names: list[str], date: dt.datetime) -> np.ndarray:
        """ Get spot prices for specified names on given date """
        spots = [self.get_spot(name, date) for name in names]
        return np.asarray(spots)

    def get_spot_data(self, name: str, date: dt.datetime) -> SpotData:
        """ Retrieve spot data object """
        folder = self.root / 'spot'
        file = Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + ".json")
        return spot_mod.spotdata_from_file(file)

    def get_eq_forward_data(self, name: str, date: dt.datetime) -> EqForwardData:
        """ Retrieve EQ forward data object """
        folder = self.root / 'eqforwards'
        file = Path(folder) / name / (date.strftime(dts.DATE_FILE_FORMAT) + ".json")
        return eqfwd.eqforwarddata_from_file(file)

    def get_eq_vol_data(self, name: str, date: dt.datetime) -> EqVolSurfaceData:
        """ Retrieve EQ vol surface data object """
        folder = self.root / 'eqoptions'
        file = eqvol.data_file(name, date, folder=folder)
        return eqvol.eqvolsurfacedata_from_file(file)


def get_eq_forward_curves(names: list[str], date: dt.datetime,
                          provider: MarketDataProvider) -> list[EqForwardCurve]:
    """ Retrieve EQ forward curves """
    spots = provider.get_spots(names, date)

    fwd_curves = []
    for name, spot_ in zip(names, spots, strict=True):
        data = provider.get_eq_forward_data(name, date)
        curve = EqForwardCurve(valdate=date, interp_var='forward', interp_type='cubicspline')
        curve.calibrate(data, spot_)
        fwd_curves.append(curve)

    return fwd_curves


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    md_provider = MarketDataFileProvider()
    obj = md_provider.get_yieldcurve("USD.SOFR.1D", valdate)
    print(obj)
