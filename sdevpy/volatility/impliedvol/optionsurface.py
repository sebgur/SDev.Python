import logging
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sdevpy.analytics import black, bachelier
from sdevpy.maths import metrics
from sdevpy.utilities.tools import isequal
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
                        vol_surface: list[list[float]], option_type: str='straddle', voltol: float=1e-4):
    """ Prepare surface of targets for calibration with estimate of tolerance """
    cf_price_surface = []
    ftols = []
    for exp_idx, expiry in enumerate(expiries):
        fwd = fwds[exp_idx]
        strikes = np.asarray(strike_surface[exp_idx])
        vols = np.asarray(vol_surface[exp_idx])
        if option_type.lower() == 'call':
            cf_price = black.price(expiry, strikes, True, fwd, vols)
            cf_price_bump = black.price(expiry, strikes, True, fwd, vols + voltol)
        elif option_type.lower() == 'put':
            cf_price = black.price(expiry, strikes, False, fwd, vols)
            cf_price_bump = black.price(expiry, strikes, False, fwd, vols + voltol)
        else: # Assumed straddle
            cf_price = black.price(expiry, strikes, True, fwd, vols)
            cf_price_bump = black.price(expiry, strikes, True, fwd, vols + voltol)
            cf_price += black.price(expiry, strikes, False, fwd, vols)
            cf_price_bump += black.price(expiry, strikes, False, fwd, vols + voltol)

        cf_price_surface.append(cf_price)
        # vols = vols + itol
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


def convert_target(target: OptionTarget, to_type: OptionQuoteType, to_shift: float) -> OptionTarget:
    """ Convert option target from its type to the chosen target type """
    t, f, k, is_call = target.expiry, target.forward, target.strike, target.is_call
    from_input, from_type, from_shift = target.market_input, target.quote_type, target.market_shift
    conv_target = OptionTarget(expiry=t, forward=f, strike=k, market_input=0.0, is_call=is_call,
                               quote_type=to_type, market_shift=to_shift, is_atm=target.is_atm)
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
                    fwd_premium = black.price(expiry, strike + from_shift, is_call, fwd + from_shift, vol)
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
                    to_input = bachelier.implied_vol_jaeckel(expiry, strike, is_call, fwd, fwd_premium)
                case OptionQuoteType.ShiftedLogNormalVol:
                    to_input = black.implied_vol(expiry, strike + to_shift, is_call, fwd + to_shift, fwd_premium)
                case _:
                    raise ValueError(f"Unknown quotation type: {to_type}")

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
        else:
            conv_options.append(conv_at_exp)

    return conv_options


def plot_transform_surface(expiries: npt.ArrayLike, strikes: npt.ArrayLike, are_calls: npt.ArrayLike, fwd: float,
                           ref_prices: npt.ArrayLike, mod_prices: npt.ArrayLike, title_: str,
                           transform: str='ShiftedBlackScholes', ref_name: str='Reference',
                           mod_name: str='Model'):
    """ Calculate quantities to display for the surface and display them in charts. Transformed
        quantities available are: Price, ShiftedBlackScholes (3%) and Bachelier (normal vols). """
    # Transform prices
    ref_disp = transform_surface(expiries, strikes, are_calls, fwd, ref_prices, transform)
    mod_disp = transform_surface(expiries, strikes, are_calls, fwd, mod_prices, transform)

    # Display transformed prices
    num_charts = expiries.shape[0]
    num_cols = 2
    num_rows = int(num_charts / num_cols)
    ylabel = 'Price' if transform == 'Price' else 'Vol'

    fig, axs = plt.subplots(num_rows, num_cols, layout="constrained")
    fig.suptitle(title_, size='x-large', weight='bold')
    fig.set_size_inches(12, 8)
    for i in range(num_rows):
        for j in range(num_cols):
            k = num_cols * i + j
            axs[i, j].plot(strikes[k], ref_disp[k], color='blue', label=ref_name)
            axs[i, j].plot(strikes[k], mod_disp[k], color='red', label=mod_name)
            axs[i, j].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
            axs[i, j].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
            axs[i, j].set_xlabel('Strike')
            axs[i, j].set_ylabel(ylabel)
            axs[i, j].set_title(f"T={expiries[k, 0]}")
            axs[i, j].legend(loc='upper right')

    plt.show()


def transform_surface(expiries: npt.ArrayLike, strikes: npt.ArrayLike, are_calls: npt.ArrayLike, fwd: float,
                      prices: npt.ArrayLike, transform: str='ShiftedBlackScholes'):
    """ Tranform prices into: Price, ShiftedBlackScholes (3%) and Bachelier (normal vols). """
    # Transform prices
    # trans_prices = []
    num_expiries = expiries.shape[0]
    num_strikes = strikes.shape[1]
    if transform == 'Price':
        trans_prices = prices
    elif transform == 'ShiftedBlackScholes':
        trans_prices = np.ndarray(shape=(num_expiries, num_strikes))
        shift = 0.03
        sfwd = fwd + shift
        for i, expiry in enumerate(expiries):
            strikes_ = strikes[i]
            are_calls_ = are_calls[i]
            # trans_prices_ = []
            for j, strike in enumerate(strikes_):
                sstrike = strike + shift
                trans_prices[i, j] = black.implied_vol(expiry, sstrike, are_calls_[j], sfwd,
                                                       prices[i, j])
                # trans_prices_.append(black.implied_vol(expiry, sstrike, are_calls_[j], sfwd,
                #                                        prices[i, j]))
            # trans_prices.append(trans_prices_)
    elif transform == 'Bachelier':
        trans_prices = np.ndarray(shape=(num_expiries, num_strikes))
        for i, expiry in enumerate(expiries):
            strikes_ = strikes[i]
            are_calls_ = are_calls[i]
            # trans_prices_ = []
            for j, strike in enumerate(strikes_):
                trans_prices[i, j] = bachelier.implied_vol_jaeckel(expiry, strike, are_calls_[j], fwd,
                                                           prices[i, j])
            #     trans_prices_.append(bachelier.implied_vol(expiry, strike, are_calls_[j], fwd,
            #                                                prices[i, j]))
            # trans_prices.append(trans_prices_)
    else:
        raise ValueError("Unknown transform type: " + transform)

    return trans_prices
    # return np.asarray(trans_prices)
