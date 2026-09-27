import logging
import datetime as dt
import numpy as np
from pathlib import Path
from sdevpy.utilities import dates as dts
from sdevpy.market import spot as spot_mod
from sdevpy.market import correlations
from sdevpy.market.eq import eqforward as eqfwd
from sdevpy.market.eq import eqvolsurface as eqvol
from sdevpy.market.eq.eqforward import EqForwardData
from sdevpy.market.eq.eqvolsurface import EqVolSurfaceData
from sdevpy.market.fx import fxvolsurface as fxvol
from sdevpy.market import fixings
from sdevpy.market.spot import SpotData
from sdevpy.market.fx.fxvolsurface import FxVolSurfaceData
from sdevpy.market.fixings import FixingHandler
from sdevpy.market.source import MarketDataSource
from sdevpy import datapaths
log = logging.getLogger(__name__)


class MarketDataFileSource(MarketDataSource):
    """ Reads market data DTOs from files on disk """
    def __init__(self, root: str|Path=None):
        if root is None:
            self.root = datapaths.marketdata_path()
            log.info(f"No root given, using default data folder: {self.root}")
        else:
            self.root = Path(root)

    def get_fixing_handler(self, name: str, interpolate: bool=False) -> FixingHandler:
        """ Retrieve fixings handler """
        folder = self.root / 'fixings'
        return fixings.fixinghandler(name, interpolate=interpolate, folder=folder)

    def get_correlations(self, names: list[str], date: dt.datetime) -> np.ndarray:
        """ Retrieve correlations """
        folder = self.root / 'correlations'
        return correlations.get_correlations(names, date, folder=folder)

    def get_spot_data(self, name: str, date: dt.datetime) -> SpotData:
        """ Retrieve spot data object """
        return spot_mod.spotdata_from_file(self._data_file('spot', name, date))

    def get_eq_forward_data(self, name: str, date: dt.datetime) -> EqForwardData:
        """ Retrieve EQ forward data object """
        return eqfwd.eqforwarddata_from_file(self._data_file('eqforwards', name, date))

    def get_eq_vol_data(self, name: str, date: dt.datetime) -> EqVolSurfaceData:
        """ Retrieve EQ vol surface data object """
        return eqvol.eqvolsurfacedata_from_file(self._data_file('eqoptions', name, date))

    def get_fx_vol_data(self, pair: str, date: dt.datetime) -> FxVolSurfaceData:
        """ Retrieve FX vol surface data object """
        return fxvol.fxvolsurfacedata_from_file(self._data_file('fxoptions', pair, date))

    def _data_file(self, category: str, name: str, date: dt.datetime) -> Path:
        """ Data file for given category, name and date: root/category/name/yyyymmdd.json """
        return self.root / category / name / (date.strftime(dts.DATE_FILE_FORMAT) + ".json")


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
