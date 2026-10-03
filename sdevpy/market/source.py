import datetime as dt
import numpy as np
from abc import ABC, abstractmethod
from sdevpy.market.spot import SpotData
from sdevpy.market.eq.eqforwarddata import EqForwardData
from sdevpy.market.eq.eqvoldata import EqVolData
from sdevpy.market.fx.fxvoldata import FxVolData
from sdevpy.market.fixings import FixingHandler


class MarketDataSource(ABC):
    """ Raw access to a market data store. Returns Data Transfer Objects (DTOs). No caching, no derived objects. """
    @abstractmethod
    def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler: ...

    @abstractmethod
    def get_correlations(self, names: list[str], date: dt.datetime) -> np.ndarray: ...

    @abstractmethod
    def get_spotdata(self, name: str, date: dt.datetime) -> SpotData: ...

    @abstractmethod
    def get_eqforwarddata(self, name: str, date: dt.datetime) -> EqForwardData: ...

    @abstractmethod
    def get_eqvoldata(self, name: str, date: dt.datetime) -> EqVolData: ...

    @abstractmethod
    def get_fxvoldata(self, pair: str, date: dt.datetime) -> FxVolData: ...
