import datetime as dt
import numpy.typing as npt
from sdevpy.calibration.rates.yieldcurve import YieldCurve
from sdevpy.conventions.fxdates import fx_spot_date


class FxForwardCurve:
    def __init__(self, valdate: dt.datetime, forccy: str, domccy: str):
        self.valdate = valdate
        self.forccy, self.domccy = forccy, domccy
        self.spot0 = None
        self.domcurve, self.forcurve = None, None

    def load_calibrated(self, spot: float, forcurve: YieldCurve, domcurve: YieldCurve) -> None:
        """ Given already curves and spot, imply the t=0 spot equivalent """
        sdate = self.spot_date()
        self.forcurve = forcurve
        self.domcurve = domcurve
        # Imply spot at t = 0
        self.spot0 = spot / self.forcurve.discount(sdate) * self.domcurve.discount(sdate)

    def value(self, date: dt.datetime|list[dt.datetime]) -> npt.ArrayLike:
        return self.spot0 * self.forcurve.discount(date) / self.domcurve.discount(date)

    def value_float(self, t) -> npt.ArrayLike:
        return self.spot0 * self.forcurve.discount_float(t) / self.domcurve.discount_float(t)

    def spot_date(self) -> dt.datetime:
        """ Calculate spot date corresponding to valdate """
        return fx_spot_date(self.valdate, self.forccy, self.domccy)


if __name__ == "__main__":
    print("Hello")
