import copy
import logging
import datetime as dt
from collections.abc import Callable, Hashable
from sdevpy.conventions import fxconventions
from sdevpy.calibration.source import CalibrationDataSource
from sdevpy.calibration.rates.yieldcurve import YieldCurve
from sdevpy.market.dataset import MarketDataSet
from sdevpy.market.fx.fxforward import FxForwardCurve
from sdevpy.volatility.impliedvol import impliedvol as iv_mod
from sdevpy.volatility.impliedvol import impliedvol_factory as ivf
from sdevpy.volatility.localvol import localvol as lv_mod
from sdevpy.volatility.localvol import localvol_factory as lvf
log = logging.getLogger(__name__)


name_model_map = {'ABC': 'BiExp', 'KLM': 'VSVI', 'XYZ': 'Matrix'}
RFR_CURVES = {fxconventions.USD: 'USD.SOFR.1D'}


class CalibrationDataSet:
    """ Calibration data for a single datetime.

        Raw data (yield curves, implied/local/FX vol data dicts) is loaded lazily from the source
        and cached on tuple[str='kind', key]. Missing data is cached as None.

        Vol model objects are mutable (calibrated in place) and depend on t_grid, so they are
        never cached: they are rebuilt from a copy of the cached data on every call.

        Saving writes through to the source and drops the cached entry, so the next read
        reloads exactly what was stored. """
    def __init__(self, date: dt.datetime, source: CalibrationDataSource):
        self.date = date
        self._source = source
        self._cache: dict[tuple[str, Hashable], object] = {}

    def _get(self, kind: str, key: Hashable, loader: Callable[[], object]):
        ckey = (kind, key)
        if ckey not in self._cache:
            self._cache[ckey] = loader()
        return self._cache[ckey]

    # Retrieve raw data (cached)
    def get_yieldcurve(self, name: str) -> YieldCurve:
        return self._get('yc', name, lambda: self._source.get_yieldcurve(name, self.date))

    def get_impliedvol_data(self, name: str, model_name: str) -> dict|None:
        return self._get('ivdata', (name, model_name),
                         lambda: self._source.get_impliedvol_data(name, self.date, model_name))

    def get_localvol_data(self, name: str, model_name: str) -> dict|None:
        return self._get('lvdata', (name, model_name),
                         lambda: self._source.get_localvol_data(name, self.date, model_name))

    def get_fxvol_data(self, pair: str) -> dict|None:
        return self._get('fxvoldata', pair, lambda: self._source.get_fxvol_data(pair, self.date))

    # Retrieve curves
    def get_rfrcurve(self, ccy: str) -> YieldCurve:
        """ Get RFR curve in the specified currency """
        if ccy not in RFR_CURVES:
            raise ValueError(f"Currency not set in RFR curve map: {ccy}")
        return self.get_yieldcurve(RFR_CURVES[ccy])

    def get_xccycurve(self, ccy: str) -> YieldCurve:
        """ Get cross-currency curve to USD for given ccy. Return USD RFR curve if ccy = USD.
            By enforced convention, the cross-currency curve to USD for e.g. EUR must be EUR.XCCY. """
        if ccy == fxconventions.USD:
            log.debug('Requested USD xccy curve: effectively USD RFR')
            return self.get_rfrcurve(fxconventions.USD)
        curve_id = f"{ccy}.XCCY"
        log.debug(f'Requested {ccy} xccy curve: effectively {curve_id}')
        return self.get_yieldcurve(curve_id)

    def get_fx_forward_curve(self, pair: str, mkt: MarketDataSet) -> FxForwardCurve:
        """ FX forward curve. Not cached: it depends on mkt, which may be a bumped dataset """
        if mkt.date != self.date:
            raise ValueError(f"Market date {mkt.date} does not match calibration date {self.date}")
        forccy, domccy = fxconventions.parse_fx_pair(pair)
        curve = FxForwardCurve(self.date, forccy, domccy)
        curve.load_calibrated(mkt.get_fx_spot(forccy, domccy),
                              self.get_xccycurve(forccy), self.get_xccycurve(domccy))
        return curve

    # Build vol models (not cached)
    def get_impliedvol(self, name: str, model_name: str) -> iv_mod.ImpliedVol:
        data = self.get_impliedvol_data(name, model_name)
        return ivf.get_impliedvol_from_data(copy.deepcopy(data))

    def get_localvol(self, name: str, model_name: str, t_grid: list[float]=None) -> lv_mod.LocalVol:
        data = self.get_localvol_data(name, model_name)
        return lvf.get_localvol_from_data(copy.deepcopy(data), t_grid)

    def get_localvol_or_new(self, name: str, model_name: str|None, t_grid: list[float]=None,
                            force_new: bool=False) -> lv_mod.LocalVol:
        """ Retrieve local vol for given name and model name.
            If the model name is None, we infer it from the (name, model) map.
            t_grid, if given, is used to define the time grid of the model, interpolating
            from the stored grid if a stored model is present.
            If t_grid is not given and there is a stored model, the grid of the stored model
            is used. If t_grid is not given and there is no stored model, an error is thrown.
            Args:
                - force_new: return new local vol even if there is an existing one
        """
        model_name = (name_model_map.get(name, None) if model_name is None else model_name)
        if model_name is None:
            raise ValueError(f"No model name specified for name: {name}")

        if not force_new and self.get_localvol_data(name, model_name) is not None:
            return self.get_localvol(name, model_name, t_grid)

        if t_grid is None:
            msg = f"No stored local vol for {name} on {self.date:%Y-%m-%d}"
            raise ValueError(f"{msg} and no t_grid given to build a new one")

        log.info(f"Initializing new LV for {name}")
        return lvf.get_localvol_new(t_grid, model_name)

    def get_local_vols(self, names: list[str], **kwargs) -> list[lv_mod.LocalVol]:
        """ Retrieve local vols assuming calibration has already been done """
        lv_map = kwargs.get('lv_map', None)
        if lv_map is None:
            model_name = kwargs.get('model_name', None)
            return [self.get_localvol_or_new(name, model_name) for name in names]

        lvs = []
        for name in names:
            name_lv = lv_map.get(name, None)
            if name_lv is None:
                raise ValueError(f"Could not find LV object in map for name: {name}")
            lvs.append(name_lv)
        return lvs

    # Save (write-through, drop cached entry)
    def save_impliedvol(self, name: str, model_name: str, ivol: iv_mod.ImpliedVol) -> None:
        self.save_impliedvol_data(name, model_name, ivol.dump_data())

    def save_impliedvol_data(self, name: str, model_name: str, data: dict) -> None:
        self._source.save_impliedvol_data(name, self.date, model_name, data)
        self._cache.pop(('ivdata', (name, model_name)), None)

    def save_localvol(self, name: str, model_name: str, lv: lv_mod.LocalVol) -> None:
        self.save_localvol_data(name, model_name, lv.dump_data())

    def save_localvol_data(self, name: str, model_name: str, data: dict) -> None:
        self._source.save_localvol_data(name, self.date, model_name, data)
        self._cache.pop(('lvdata', (name, model_name)), None)

    def save_fxvol_data(self, pair: str, data: dict) -> None:
        self._source.save_fxvol_data(pair, self.date, data)
        self._cache.pop(('fxvoldata', pair), None)

    def clear(self):
        self._cache.clear()
