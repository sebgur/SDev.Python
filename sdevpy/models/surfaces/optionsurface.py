from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from sdevpy.analytics import black, bachelier
from sdevpy.maths import metrics
from sdevpy.tools.utils import isequal
log = logging.getLogger(__name__)

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
    market_input: float
    market_shift: float
    is_call: bool = True
    quote_type: OptionQuoteType = OptionQuoteType.LogNormalVol
    is_atm: bool = True


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


# ToDo
def convert_target(target: OptionTarget, to_type: OptionQuoteType, to_shift: float) -> OptionTarget:
    """ Convert option target from its type to the chosen target type """
    t, f, k, is_call = target.expiry, target.forward, target.strike, target.is_call
    from_input, from_type, from_shift = target.market_input, target.quote_type, target.market_shift
    conv_target = OptionTarget(expiry=t, forward=f, strike=k, is_call=is_call, quote_type=to_type,
                               market_shift=to_shift, is_atm=target.is_atm)
    conv_target.market_input = convert_option(t, k, is_call, f, from_input, from_type, from_shift,
                                              to_type, to_shift)
    return conv_target


def check_model(fwd: float, strike: float, quote_type: OptionQuoteType, shift: float) -> None:
    if quote_type == OptionQuoteType.ShiftedLogNormalVol:
        fwd += shift
        strike += shift

    if quote_type == OptionQuoteType.LogNormalVol or quote_type == OptionQuoteType.ShiftedLogNormalVol:
        if fwd < 0.0:
            raise ValueError(f"Negative forward not admissible for quote type {quote_type}")

        if strike < 0.0:
            raise ValueError(f"Negative strike not admissible for quote type {quote_type}")


def convert_option(expiry: float, strike: float, is_call: bool, fwd: float, from_input: float,
                   from_type: OptionQuoteType, from_shift: float, to_type: OptionQuoteType, to_shift: float) -> float:
    """ Convert option quote from a type to another type """
    if from_type == to_type and isequal(from_shift, to_shift):
        to_input = from_input
    else:
        check_model(fwd, strike, from_type, from_shift)
        check_model(fwd, strike, to_type, to_shift)

        # Calculate forward premium
        if from_type == OptionQuoteType.ForwardPremium:
            fwd_premium = from_input
        else:
            vol = from_input
            match from_type:
                case OptionQuoteType.LogNormalVol:
                    fwd_premium = black.price(expiry, strike, is_call, fwd, vol)
                case OptionQuoteType.NormalVol:
                    fwd_premium = bachelier.price(expiry, strike, is_call, fwd, vol)
                case OptionQuoteType.ShiftedLogNormalVol:
                    fwd_premium = black.price(expiry, strike + from_shift, is_call, fwd, vol)
                case _:
                    raise ValueError(f"Unknown quotation type: {from_type}")

        # Convert forward premium
        if to_type == OptionQuoteType.ForwardPremium:
            to_input = fwd_premium
        else:
            match to_type:
                case OptionQuoteType.LogNormalVol:
                    to_input = black.implied_vol(expiry, strike, is_call, fwd, fwd_premium)
                case OptionQuoteType.NormalVol:
                    to_input = bachelier.implied_vol(expiry, strike, is_call, fwd_premium)
                case OptionQuoteType.ShiftedLogNormalVol:
                    to_input = black.implied_vol(expiry, strike + to_shift, is_call, fwd + to_shift, fwd_premium)
                case _:
                    raise ValueError(f"Unknown quotation type: {from_type}")

    return to_input


def convert_to_target_values(target_options: list[list[OptionTarget]], target_type: OptionQuoteType, shift: float):
    """ Return target options converted to chosen type """
    conv_options = []
    for targets_at_exp in target_options:
        conv_at_exp = []
        for target in targets_at_exp:
            try:
                conv_at_exp.append(convert_target(target, target_type, shift))
            except (ValueError, RuntimeError):
                log.warning(f"Failed conversion, skipping target at expiry: {target.expiry}")

        if len(conv_at_exp) == 0: # Or we could raise ValueError
            log.warning("No successful option conversion: skipping expiry")

        conv_options.append(conv_at_exp)

    return conv_options
