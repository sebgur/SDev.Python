import datetime as dt
from dataclasses import dataclass
from sdevpy.market.repository import MarketDataRepository
from sdevpy.market.filesource import MarketDataFileSource
from sdevpy.market.dataset import MarketDataSet
from sdevpy.calibration.repository import CalibrationDataRepository
from sdevpy.calibration.filesource import CalibrationDataFileSource
from sdevpy.calibration.dataset import CalibrationDataSet


@dataclass
class PricingContext:
    market_repo: MarketDataRepository
    calib_repo: CalibrationDataRepository

    def at(self, date: dt.datetime) -> tuple[MarketDataSet, CalibrationDataSet]:
        """ Market and calibration datasets for the same date """
        return self.market_repo[date], self.calib_repo[date]


def default_market_repository() -> MarketDataRepository:
    """ File-based market data repository on the default data folder """
    return MarketDataRepository(MarketDataFileSource())


def default_calibration_repository() -> CalibrationDataRepository:
    """ File-based calibration data repository on the default data folder """
    return CalibrationDataRepository(CalibrationDataFileSource())


def default_pricing_context() -> PricingContext:
    """ Default pricing context: market and calibration are file based """
    return PricingContext(market_repo=default_market_repository(),
                          calib_repo=default_calibration_repository())
