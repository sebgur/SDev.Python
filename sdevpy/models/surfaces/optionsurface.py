from dataclasses import dataclass
from enum import Enum
from sdevpy.analytics import black
from sdevpy.maths import metrics


IS_CALL = True


class OptionQuotationType(Enum):
    LogNormalVol = 0
    NormalVol = 1
    ShiftedLogNormalVol = 2
    ForwardPremium = 3


@dataclass
class OptionTarget:
    expiry: float
    forward: float
    strike: float
    is_call: bool = True
    is_atm: bool
    market_input: float
    quote_type: OptionQuotationType = OptionQuotationType.LogNormalVol
    market_shift: float


# class OptionSurface:
#     def __init__(self):
#         self.expiry_dates = []


def calibration_targets(expiries: list[float], fwds: list[float], strike_surface: list[list[float]],
                        vol_surface: list[list[float]]):
    """ Prepare surface of targets for calibration with estimate of tolerance """
    cf_price_surface = []
    ftols = []
    itol = 1e-6 # 1bp
    for exp_idx, expiry in enumerate(expiries):
        fwd = fwds[exp_idx]
        strikes = strike_surface[exp_idx]
        vols = vol_surface[exp_idx]
        cf_price = black.price(expiry, strikes, IS_CALL, fwd, vols)
        cf_price_surface.append(cf_price)
        vols = vols + itol
        cf_price_bump = black.price(expiry, strikes, IS_CALL, fwd, vols)
        ftols.append(metrics.rmse(cf_price, cf_price_bump))

    return cf_price_surface, ftols


def calibration_targets(expiries, forwards) -> list[list[OptionTarget]]:
    """ Return calibration targets """
    file = vsurf.data_file(vsurf.test_data_folder(), name, valdate)
    surface_data = vsurf.eqvolsurfacedata_from_file(file)
    expiries = surface_data.expiries
    fwds = surface_data.forwards
    strike_surface = surface_data.get_strikes('absolute')
    vol_surface = surface_data.vols
