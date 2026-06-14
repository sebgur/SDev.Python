from dataclasses import dataclass
from sdevpy.market.provider import MarketDataProvider, MarketDataFileProvider
from sdevpy.calibration.provider import CalibrationDataProvider
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider


@dataclass
class PricingContext:
    market_provider: MarketDataProvider
    calib_provider: CalibrationDataProvider


def default_pricing_context() -> PricingContext:
    """ Default pricing context: both market data and calibration data providers
        are file based and their root is the corresponding test folders. """
    return PricingContext(market_provider=MarketDataFileProvider(),
                          calib_provider=CalibrationDataFileProvider())
