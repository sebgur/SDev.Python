from dataclasses import dataclass
from sdevpy.market.repository import MarketDataRepository
from sdevpy.market.filesource import MarketDataFileSource
# from sdevpy.market.provider import MarketDataProvider
# from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.calibration.provider import CalibrationDataProvider
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider


@dataclass
class PricingContext:
    market_repo: MarketDataRepository
    # market_provider: MarketDataProvider
    calib_provider: CalibrationDataProvider


def default_market_repository() -> MarketDataRepository:
    """ File-based market data repository on the default data folder """
    return MarketDataRepository(MarketDataFileSource())


def default_pricing_context() -> PricingContext:
    """ Default pricing context: market andd calibration are file based """
    return PricingContext(market_repo=default_market_repository(),
                          calib_provider=CalibrationDataFileProvider())
