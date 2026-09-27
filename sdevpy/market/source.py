import datetime as dt
import numpy as np
from abc import ABC, abstractmethod
from sdevpy.market.spot import SpotData
from sdevpy.market.eq.eqforward import EqForwardData
from sdevpy.market.eq.eqvolsurface import EqVolSurfaceData
from sdevpy.market.fx.fxvolsurface import FxVolSurfaceData
from sdevpy.market.fixings import FixingHandler


class MarketDataSource(ABC):
    """ Raw access to a market data store. Returns Data Transfer Objects (DTOs). No caching, no derived objects. """
    @abstractmethod
    def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler: ...

    @abstractmethod
    def get_correlations(self, names: list[str], date: dt.datetime) -> np.ndarray: ...

    @abstractmethod
    def get_spot_data(self, name: str, date: dt.datetime) -> SpotData: ...

    @abstractmethod
    def get_eq_forward_data(self, name: str, date: dt.datetime) -> EqForwardData: ...

    @abstractmethod
    def get_eq_vol_data(self, name: str, date: dt.datetime) -> EqVolSurfaceData: ...

    @abstractmethod
    def get_fx_vol_data(self, pair: str, date: dt.datetime) -> FxVolSurfaceData: ...
