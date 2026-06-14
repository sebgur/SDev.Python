import datetime as dt
import numpy as np
from typing import Protocol, runtime_checkable
from sdevpy.market.yieldcurve import YieldCurve
from sdevpy.market.spot import SpotData
from sdevpy.market.eqforward import EqForwardData, EqForwardCurve
from sdevpy.market.eqvolsurface import EqVolSurfaceData
from sdevpy.market.fixings import FixingHandler


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
