import datetime as dt
import numpy as np
from collections.abc import Callable, Hashable
from sdevpy.market.source import MarketDataSource
from sdevpy.market.spot import SpotData
from sdevpy.market.eqforward import EqForwardData, EqForwardCurve
from sdevpy.market.eqvolsurface import EqVolSurfaceData
from sdevpy.market.fx.fxvolsurface import FxVolSurfaceData
from sdevpy.market.fx import fxconventions


class MarketDataSet:
    """ Market data for a single datetime, giving access to the loaded sources providing the data objects (DTO).

        The DTOs are loaded lazily from the source and cached on tuple[str='kind', key='name']:
            - 'kind' is an object type (say EQ forward, EQ vol, etc.)
            - 'key' is the identifier/name of the object (say SPX, etc.).

        RAW_KINDS can be bumped/overriden. Non-RAW_KINDS are rebuilt after bump. """
    RAW_KINDS = {'spot', 'eqfwd', 'eqvol', 'fxvol', 'corr'}

    def __init__(self, date: dt.datetime, source: MarketDataSource, base: 'MarketDataSet|None'=None):
        """ Arg 'base' not supposed to be passed from outside, but only internally by with_overrides() """
        self.date = date
        self._source = source
        self._base = base
        self._cache: dict[tuple[str, Hashable], object] = {}

    def _get(self, kind: str, key: Hashable, loader: Callable[[], object]):
        ckey = (kind, key)
        if ckey not in self._cache:
            if self._base is not None and kind in self.RAW_KINDS:
                self._cache[ckey] = self._base._get(kind, key, loader)
            else:
                self._cache[ckey] = loader()
        return self._cache[ckey]

    def with_overrides(self, overrides: dict[tuple[str, Hashable], object]) -> 'MarketDataSet':
        """ New dataset where the given raw DTOs are replaced. Other raw DTOs are shared with this
            dataset, derived objects (curves, FX crosses) are rebuilt from the overridden data.
            The bumped set will build its derived types from the raw DTOs of the base and the
            bumped DTOs it received as overrides. """
        bumped = MarketDataSet(self.date, self._source, base=self)
        bumped._cache.update(overrides)
        return bumped

    # Retrieve DTOs (cached)
    def get_spot_data(self, name: str) -> SpotData:
        return self._get('spot', name, lambda: self._source.get_spot_data(name, self.date))

    def get_eq_forward_data(self, name: str) -> EqForwardData:
        return self._get('eqfwd', name, lambda: self._source.get_eq_forward_data(name, self.date))

    def get_eq_vol_data(self, name: str) -> EqVolSurfaceData:
        return self._get('eqvol', name, lambda: self._source.get_eq_vol_data(name, self.date))

    def get_fx_vol_data(self, pair: str) -> FxVolSurfaceData:
        return self._get('fxvol', pair, lambda: self._source.get_fx_vol_data(pair, self.date))

    def get_correlations(self, names: list[str]) -> np.ndarray:
        return self._get('corr', tuple(names), lambda: self._source.get_correlations(names, self.date))

    # Retrieve derived objects (cached)
    def get_spot(self, name: str) -> float:
        return self.get_spot_data(name).value

    def get_spots(self, names: list[str]) -> np.ndarray:
        return np.asarray([self.get_spot(n) for n in names])

    def get_eq_forward_curve(self, name: str) -> EqForwardCurve:
        def build():
            curve = EqForwardCurve(valdate=self.date, interp_var='forward', interp_type='cubicspline')
            curve.calibrate(self.get_eq_forward_data(name), self.get_spot(name))
            return curve
        return self._get('eqfwdcurve', name, build)

    def get_eq_forward_curves(self, names: list[str]) -> list[EqForwardCurve]:
        return [self.get_eq_forward_curve(n) for n in names]

    def get_fx_spot(self, forccy: str, domccy: str) -> float:
        if forccy == domccy:
            return 1.0
        return self._usd_leg_spot(forccy) / self._usd_leg_spot(domccy)

    def _usd_leg_spot(self, ccy: str) -> float:
        if ccy == fxconventions.USD:
            return 1.0
        conv_for, conv_dom = fxconventions.usd_conventional_pair(ccy)
        quote = self.get_spot(conv_for + conv_dom)
        return quote if conv_for == ccy else 1.0 / quote

    def clear(self):
        self._cache.clear()
