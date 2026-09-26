import logging
from abc import ABC, abstractmethod
import datetime as dt
from sdevpy.market.yieldcurve import YieldCurve
from sdevpy.market.provider import MarketDataProvider
from sdevpy.market.fx import fxconventions
from sdevpy.market.fx.fxforward import FxForwardCurve
from sdevpy.volatility.impliedvol import impliedvol as iv_mod
from sdevpy.volatility.impliedvol import impliedvol_factory as ivf
from sdevpy.volatility.localvol import localvol as lv_mod
from sdevpy.volatility.localvol import localvol_factory as lvf
log = logging.getLogger(__name__)


name_model_map = {'ABC': 'BiExp', 'KLM': 'VSVI', 'XYZ': 'Matrix'}
RFR_CURVES = {fxconventions.USD: 'USD.SOFR.1D'}


class CalibrationDataProvider(ABC):
    @abstractmethod
    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve: ...

    @abstractmethod
    def get_impliedvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None: ...

    @abstractmethod
    def get_localvol_data(self, name: str, date: dt.datetime, model_name: str) -> dict|None: ...

    def get_rfrcurve(self, ccy: str, date: dt.datetime) -> YieldCurve:
        """ Get RFR curve in the specified currency """
        if ccy in RFR_CURVES:
            return self.get_yieldcurve(RFR_CURVES[ccy], date)
        else:
            raise ValueError(f"Currency not set in RFR curve map: {ccy}")

    def get_xccycurve(self, ccy: str, date: dt.datetime) -> YieldCurve:
        """ Get cross-currency curve to USD for given ccy. Return USD RFR curve if ccy = USD.
            By enforced convention, the cross-currency curve to USD for e.g. EUR must be EUR.XCCY. """
        if ccy == fxconventions.USD:
            log.debug('Requested USD xccy curve: effectively USD RFR')
            return self.get_rfrcurve(fxconventions.USD, date)
        else:
            curve_id = f"{ccy}.XCCY"
            log.debug(f'Requested {ccy} xccy curve: effectively {curve_id}')
            return self.get_yieldcurve(curve_id, date)

    def get_fx_forward_curve(self, name: str, date: dt.datetime, md_prov: MarketDataProvider) -> FxForwardCurve:
        """ Retrieve FX forward curves """
        forccy, domccy = fxconventions.parse_fx_pair(name)
        spot = md_prov.get_fx_spot(forccy, domccy, date)
        forcurve = self.get_xccycurve(forccy, date)
        domcurve = self.get_xccycurve(domccy, date)

        curve = FxForwardCurve(date, forccy, domccy)
        curve.load_calibrated(spot, forcurve, domcurve)
        return curve

    def get_impliedvol(self, name: str, date: dt.datetime, model_name: str) -> iv_mod.ImpliedVol:
        """ Retrieve implied vol knowing name, date and model name """
        data = self.get_impliedvol_data(name, date, model_name)
        return ivf.get_impliedvol_from_data(data)

    def get_localvol(self, name: str, date: dt.datetime, model_name: str, t_grid: list[float]=None) -> lv_mod.LocalVol:
        """ Retrieve local vol knowing name, date and model name """
        data = self.get_localvol_data(name, date, model_name)
        return lvf.get_localvol_from_data(data, t_grid)

    def get_localvol_or_new(self, name: str, date: dt.datetime, model_name: str,
                            t_grid: list[float]=None, force_new: bool=False) -> lv_mod.LocalVol:
        """ Retrieve local vol for given name, date and model name.
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

        # Look for an existing model file
        lv = None
        if not force_new: # Try to get it from calibration provider, None if absent
            data = self.get_localvol_data(name, date, model_name)
            if data is not None:
                lv = lvf.get_localvol_from_data(data, t_grid)

        if lv is None:
            if t_grid is None:
                msg = f"No stored local vol for {name} on {date:%Y-%m-%d}"
                raise ValueError(f"{msg} and no t_grid given to build a new one")

            log.info(f"Initializing new LV for {name}")
            lv = lvf.get_localvol_new(t_grid, model_name)

        return lv

    def get_local_vols(self, names: list[str], valdate: dt.datetime, **kwargs) -> list[lv_mod.LocalVol]:
        """ Retrieve local vols assuming calibration has already been done """
        lv_map = kwargs.get('lv_map', None)
        lvs = []
        if lv_map is None: # Get from CalibrationDataProvider
            model_name = kwargs.get('model_name', None)
            for name in names:
                lvs.append(self.get_localvol_or_new(name, valdate, model_name))
        else: # Read from map
            for name in names:
                name_lv = lv_map.get(name, None)
                if name_lv is not None:
                    lvs.append(name_lv)
                else:
                    raise ValueError(f"Could not find LV object in map for name: {name}")

        return lvs


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
