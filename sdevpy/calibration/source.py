import datetime as dt
from abc import ABC, abstractmethod
from sdevpy.calibration.rates.yieldcurve import YieldCurve


class CalibrationDataSource(ABC):
    """ Raw access to a calibration data store. Returns stored curves and data dictionaries.
        No caching, no model objects. """
    # Read
    @abstractmethod
    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve: ...

    @abstractmethod
    def get_impliedvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None: ...

    @abstractmethod
    def get_localvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None: ...

    @abstractmethod
    def get_fxvoldata(self, pair: str, date: dt.datetime) -> dict|None: ...

    # Write
    @abstractmethod
    def save_impliedvol_data(self, name: str, date: dt.datetime, model_name: str, data: dict) -> None: ...

    @abstractmethod
    def save_localvol_data(self, name: str, date: dt.datetime, model_name: str, data: dict) -> None: ...

    @abstractmethod
    def save_fxvoldata(self, pair: str, date: dt.datetime, data: dict) -> None: ...
