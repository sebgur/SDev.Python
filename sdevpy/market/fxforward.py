import datetime as dt
from pathlib import Path
import numpy.typing as npt
from sdevpy.utilities import dates as dts
from sdevpy.utilities import jsonmanager as jsm
from sdevpy.market.yieldcurve import YieldCurve
from sdevpy.utilities.scalendar import make_calendar


# class FxForwardData:
#     def __init__(self, valdate: str, spot_date: str, spot: float, pillars: dict, **kwargs):
#         self.valdate = valdate
#         self.snapdate = kwargs.get('snapdate', self.valdate)
#         self.name = kwargs.get('name', '')
#         self.spot_date = spot_date
#         self.spot = spot

#         pillars.sort(key=lambda x: x['expiry'])
#         self.expiries = np.asarray([p['expiry'] for p in pillars])
#         self.forwards = np.asarray([p['forward'] for p in pillars])

#     def dump(self, file: str|Path, indent: int=2):
#         """ Dump data into file """
#         data = self.dump_data()
#         jsm.serialize(data, file, indent)

#     def dump_data(self) -> dict:
#         """ Dump data to dictionary """
#         pillars = [{'expiry': e.strftime(dts.DATE_FORMAT), 'forward': f}
#                    for e, f in zip(self.expiries, self.forwards, strict=True)]
#         return {'name': self.name, 'valdate': self.valdate.strftime(dts.DATE_FORMAT),
#                 'snapdate': self.snapdate.strftime(dts.DATETIME_FORMAT),
#                 'spot_date': self.spot_date.strftime(dts.DATE_FORMAT),
#                 'spot': self.spot, 'pillars': pillars}


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

    def value(self, date: dt.datetime | list[dt.datetime]) -> npt.ArrayLike:
        return self.spot0 * self.forcurve.discount(date) / self.domcurve.discount(date)

    def value_float(self, t) -> npt.ArrayLike:
        return self.spot0 * self.forcurve.discount_float(t) / self.domcurve.discount_float(t)

    def spot_date(self) -> dt.datetime:
        """ Calculate spot date corresponding to valdate """
        return fx_spot_date(self.valdate, self.forccy, self.domccy)


FX_SPOT_LAG_OVERRIDES = {
    frozenset({'USD', 'CAD'}): 1,
    frozenset({'USD', 'KZT'}): 1,
    frozenset({'USD', 'PHP'}): 1,
    frozenset({'USD', 'TRY'}): 1,
    frozenset({'CAD', 'KZT'}): 1,
    frozenset({'CAD', 'PHP'}): 1,
    frozenset({'CAD', 'TRY'}): 1,
    frozenset({'KZT', 'PHP'}): 1,
    frozenset({'KZT', 'TRY'}): 1,
    frozenset({'PHP', 'TRY'}): 1
}
DEFAULT_SPOT_LAG_DAYS = 2


def fx_spot_lag(forccy: str, domccy: str) -> int:
    """ Business days from trade date to spot, per FX market convention """
    return FX_SPOT_LAG_OVERRIDES.get(frozenset({forccy, domccy}), DEFAULT_SPOT_LAG_DAYS)


def fx_spot_date(valdate: dt.datetime, forccy: str, domccy: str) -> dt.datetime:
    """ USD pairs walk the lag on the joint currency-pair calendar. Crosses (neither leg USD) only need
        the first hop good in the pair's own centers, then require USD too, since crosses settle via two USD legs.
    """
    if forccy == domccy:
        raise ValueError(f"Foreign and domestic currency are the same: {forccy}")

    lag = fx_spot_lag(forccy, domccy)
    pair_cal = make_calendar(f"{forccy},{domccy}")

    if 'USD' in (forccy, domccy):
        return pair_cal.add_business_days(valdate, lag)

    settle_cal = make_calendar(f"{forccy},{domccy},USD")
    date = pair_cal.add_business_days(valdate, 1)
    return settle_cal.add_business_days(date, lag - 1) if lag > 1 else date


# def fxforwarddata_from_file(file: str|Path) -> FxForwardData:
#     """ Extract FxForwardData object out of file """
#     data = jsm.deserialize(file)

#     pillars = data.get('pillars')
#     for pillar in pillars:
#         pillar['expiry'] = dt.datetime.strptime(pillar['expiry'], dts.DATE_FORMAT)

#     valdate = dt.datetime.strptime(data.get('valdate'), dts.DATE_FORMAT)
#     spot_date = dt.datetime.strptime(data.get('spot_date'), dts.DATE_FORMAT)
#     snapdate = dt.datetime.strptime(data.get('snapdate'), dts.DATETIME_FORMAT)

#     return FxForwardData(valdate, spot_date, data.get('spot'), pillars,
#                          name=data.get('name'), snapdate=snapdate)


if __name__ == "__main__":
    print("Hello")
