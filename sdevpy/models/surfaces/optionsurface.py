from dataclasses import dataclass
from enum import Enum
from sdevpy.analytics import black
from sdevpy.maths import metrics
from sdevpy.tools.utils import isequal


IS_CALL = True


class OptionQuoteType(Enum):
    LogNormalVol = 0
    NormalVol = 1
    ShiftedLogNormalVol = 2
    ForwardPremium = 3


@dataclass
class OptionTarget:
    expiry: float
    forward: float
    strike: float
    is_atm: bool
    market_input: float
    market_shift: float
    is_call: bool = True
    quote_type: OptionQuoteType = OptionQuoteType.LogNormalVol


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


def keep_positive(targets: list[list[OptionTarget]], skip_empty_expiries: bool=True) -> list[list[OptionTarget]]:
    """ Select options that have both positive forwards and strikes """
    list_array = []
    for targets_at_exp in targets:
        # Select positive
        positive = []
        for target in targets_at_exp:
            shift = 0
            f = target.forward + shift
            k = target.strike + shift
            if f > 0.0 and k > 0.0:
                positive.append(target)

        if len(positive) > 0:
            list_array.append(positive)
        elif len(positive) == 0 and not skip_empty_expiries:
            raise ValueError("No positive-rate options in calibration basket")

    return list_array


def check_expiries_and_forwards(targets: list[list[OptionTarget]]) -> None:
    """ Check expiries and forwards are consistent within expiry sections """
    for targets_at_exp in targets:
        if len(targets_at_exp) == 0:
            raise ValueError("No options found when checking expiries and forwards")

        expiry = targets_at_exp[0].expiry
        fwd = targets_at_exp[0].forward
        for target in targets_at_exp:
            if not isequal(target.expiry, expiry) or not isequal(target.forward, fwd):
                raise ValueError("Inconsistent expiries in calibration basket")


def convert_to_target_values(target_options: list[list[OptionTarget]], target_type: OptionQuoteType, shift: float):
    """ Return target options converted to chosen type """
    conv_options = []
    for targets_at_exp in target_options:
        conv_at_exp = []
        for target in targets_at_exp:
            try:
                conv_at_exp.append(target.convert(target_type, shift))
            except ValueError:
                print("Failed conversion, target not selected")

        if len(conv_at_exp) == 0: # Or we could skip this expiry by not adding it
            print("No successful option conversion at in calibration basket: skipping expiry")

        conv_options.append(conv_at_exp)

    return conv_options


# def calibration_targets(expiries, forwards) -> list[list[OptionTarget]]:
#     """ Return calibration targets """
#     file = vsurf.data_file(vsurf.test_data_folder(), name, valdate)
#     surface_data = vsurf.eqvolsurfacedata_from_file(file)
#     expiries = surface_data.expiries
#     fwds = surface_data.forwards
#     strike_surface = surface_data.get_strikes('absolute')
#     vol_surface = surface_data.vols
