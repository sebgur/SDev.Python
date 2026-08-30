from abc import ABC, abstractmethod
import datetime as dt
import numpy as np
# from typing import Protocol, runtime_checkable
from sdevpy.market.yieldcurve import YieldCurve
from sdevpy.market.spot import SpotData
from sdevpy.market.eqforward import EqForwardData, EqForwardCurve
from sdevpy.market.fxforward import FxForwardCurve
from sdevpy.market.eqvolsurface import EqVolSurfaceData
from sdevpy.market.fixings import FixingHandler
from sdevpy.market import fxspot


RFR_CURVES = {fxspot.USD: 'USD.SOFR.1D'}


# @runtime_checkable
class MarketDataProvider(ABC):
    @abstractmethod
    def get_yieldcurve(self, name: str, date: dt.datetime) -> YieldCurve: ...

    @abstractmethod
    def get_fixings(self, name: str, dates: dt.datetime|list[dt.datetime], **kwargs) -> list[float]: ...

    @abstractmethod
    def get_fixing_handler(self, name: str, **kwargs) -> FixingHandler: ...

    @abstractmethod
    def get_correlations(self, names: list[str], date: dt.datetime) -> np.ndarray: ...

    @abstractmethod
    def get_spot(self, name: str, date: dt.datetime) -> float: ...

    @abstractmethod
    def get_spots(self, names: list[str], date: dt.datetime) -> np.ndarray: ...

    @abstractmethod
    def get_spot_data(self, name: str, date: dt.datetime) -> SpotData: ...

    @abstractmethod
    def get_eq_forward_data(self, name: str, date: dt.datetime) -> EqForwardData: ...

    @abstractmethod
    def get_eq_vol_data(self, name: str, date: dt.datetime) -> EqVolSurfaceData: ...

    def get_rfrcurve(self, ccy: str, date: dt.datetime) -> YieldCurve:
        """ Get RFR curve in the specified currency """
        if ccy in RFR_CURVES:
            return self.get_yieldcurve(RFR_CURVES[ccy], date)
        else:
            raise ValueError(f"Currency not set in RFR curve map: {ccy}")

    def get_xccycurve(self, ccy: str, date: dt.datetime) -> YieldCurve:
        """ Get cross-currency curve to USD for given ccy. Return USD RFR curve if ccy = USD.
            By enforced convention, the cross-currency curve to USD for e.g. EUR must be EUR.XCCY. """
        if ccy == fxspot.USD:
            return self.get_rfrcurve(fxspot.USD, date)
        else:
            return self.get_yieldcurve(f"{ccy}.XCCY", date)

    def get_eq_forward_curves(self, names: list[str], date: dt.datetime) -> list[EqForwardCurve]:
        """ Retrieve EQ forward curves """
        spots = self.get_spots(names, date)
        fwd_curves = []
        for name, spot_ in zip(names, spots, strict=True):
            data = self.get_eq_forward_data(name, date)
            curve = EqForwardCurve(valdate=date, interp_var='forward', interp_type='cubicspline')
            curve.calibrate(data, spot_)
            fwd_curves.append(curve)

        return fwd_curves

    def get_fx_forward_curve(self, name: str, date: dt.datetime) -> FxForwardCurve:
        """ Retrieve FX forward curves """
        forccy, domccy = fxspot.parse_fx_pair(name)
        spot = self.get_fx_spot(forccy, domccy, date)
        forcurve = self.get_xccycurve(forccy, date)
        domcurve = self.get_xccycurve(domccy, date)

        curve = FxForwardCurve(date, forccy, domccy)
        curve.load_calibrated(spot, forcurve, domcurve)
        return curve

    def get_fx_spot(self, forccy: str, domccy: str, date: dt.datetime) -> float:
        """ Units of domccy per unit of forccy, e.g. get_fx_spot('EUR', 'USD', ...) ~ 1.05. USD legs are read
            from the provider in market-conventional order and inverted if the request is the other way round.
            Crosses triangulate through USD (e.g. EURJPY = EURUSD * USDJPY). """
        if forccy == domccy:
            return 1.0

        # USDs for 1 unit of forccy / USDs for 1 unit of dom ccy: so how many dom ccy for 1 unit of forccy
        return self._usd_leg_spot(forccy, date) / self._usd_leg_spot(domccy, date)

    def _usd_leg_spot(self, ccy: str, date: dt.datetime) -> float:
        """ USD value of one unit of ccy """
        if ccy == fxspot.USD:
            return 1.0

        conv_for, conv_dom = fxspot.usd_conventional_pair(ccy)
        quote = self.get_spot(conv_for + conv_dom, date)
        return quote if conv_for == ccy else 1.0 / quote



if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
