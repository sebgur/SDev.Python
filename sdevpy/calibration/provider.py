import datetime as dt
# import numpy as np
from pathlib import Path
from typing import Protocol, runtime_checkable
# from sdevpy.market import spot as spot_mod
# from sdevpy.market import correlations
# from sdevpy.market import yieldcurve as ycrv
from sdevpy.market import yieldcurve as yc
# from sdevpy.market import eqforward as eqfwd
# from sdevpy.market import eqvolsurface as eqvol
# from sdevpy.market import fixings
# from sdevpy.market.spot import SpotData
# from sdevpy.market.eqforward import EqForwardData, EqForwardCurve
# from sdevpy.market.eqvolsurface import EqVolSurfaceData
# from sdevpy.market.fixings import FixingHandler
from sdevpy.tests.test import test_calibdata_path


@runtime_checkable
class CalibrationDataProvider(Protocol):
    def get_yieldcurve(self, name: str, valdate: dt.datetime) -> yc.YieldCurve: ...
    # def get_impliedvol(self, name: str, dates: dt.datetime|list[dt.datetime], **kwargs) -> list[float]: ...
    # def get_localvol(self, name: str, **kwargs) -> FixingHandler: ...


class CalibrationDataFileProvider:
    """ Reads calibrated data from JSON/CSV files on disk """
    def __init__(self, root: str|Path=None):
        self.root = (Path(root) if root is not None else test_calibdata_path())

    def get_yieldcurve(self, name: str, date: dt.datetime) -> yc.YieldCurve:
        """ Retrieve yield curve """
        folder = str(self.root / 'yieldcurves')
        file = yc.data_file(name, date, folder)
        curve = yc.yieldcurve_from_file(file)
        return curve
        # return ycrv.get_yieldcurve(name, valdate, folder=folder)

    # def get_fixings(self, name: str, dates: dt.datetime|list[dt.datetime], **kwargs) -> list[float]:
    #     """ Retrieve fixings """
    #     handler = self.get_fixing_handler(name, **kwargs)
    #     return handler.value(dates)

    # def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler:
    #     """ Retrieve fixings handler """
    #     folder = str(self.root / 'fixings')
    #     interpolate = kwargs.get('interpolate', False)
    #     return fixings.fixinghandler(name, interpolate=interpolate, folder=folder)

    # def get_correlations(self, names: list[str], valdate: dt.datetime) -> np.ndarray:
    #     """ Retrieve correlations """
    #     folder = str(self.root / 'correlations')
    #     return correlations.get_correlations(names, valdate, folder=folder)

    # def get_spot(self, name: str, valdate: dt.datetime) -> float:
    #     """ Retrieve spot """
    #     return self.get_spot_data(name, valdate).value

    # def get_spots(self, names: list[str], valdate: dt.datetime) -> np.ndarray:
    #     """ Get spot prices for specified names on given date """
    #     spots = [] # ToDo: write as comprehension
    #     for name in names:
    #         spots.append(self.get_spot(name, valdate))

    #     return np.asarray(spots)

    # def get_spot_data(self, name: str, valdate: dt.datetime) -> SpotData:
    #     """ Retrieve spot data object """
    #     folder = str(self.root / 'spot')
    #     file = spot_mod.data_file(name, valdate, folder=folder)
    #     return spot_mod.spotdata_from_file(file)


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    # md_provider = MarketDataFileProvider()
    # obj = md_provider.get_yieldcurve("USD.SOFR.1D", valdate)
    # print(obj)
