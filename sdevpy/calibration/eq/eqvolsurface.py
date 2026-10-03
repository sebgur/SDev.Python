import numpy.typing as npt
from sdevpy.utilities import timegrids
from sdevpy.analytics import black
from sdevpy.market.eq.eqvolsurface import EqVolSurfaceData
from sdevpy.calibration.eq.eqforward import EqForwardCurve


def get_prices(vol_data: EqVolSurfaceData, fwd_curve: EqForwardCurve, option_type: str='call') -> list[npt.NDArray]:
    """ Retrieve prices """
    option_type_lw = option_type.lower()
    prices = []
    abs_strikes = get_strikes(vol_data, fwd_curve, to_type='absolute')
    for exp_idx, expiry in enumerate(vol_data.expiries):
        t = timegrids.model_time(vol_data.valdate, expiry)
        fwd = fwd_curve.value(expiry)
        strikes = abs_strikes[exp_idx]
        vols = vol_data.vols[exp_idx]
        match option_type_lw:
            case 'call':
                price = black.price(t, strikes, True, fwd, vols)
            case 'put':
                price = black.price(t, strikes, False, fwd, vols)
            case 'straddle':
                price = black.price(t, strikes, True, fwd, vols)
                price = price + black.price(t, strikes, False, fwd, vols)
            case _:
                raise ValueError(f"Invalid option type: {option_type}")

        prices.append(price)
    return prices


def get_strikes(vol_data: EqVolSurfaceData, fwd_curve: EqForwardCurve=None,
                to_type: str='absolute') -> list[npt.NDArray]:
    """ Retrieve strikes, absolute or relative """
    to_type_lw = to_type.lower()
    if to_type_lw == vol_data.strike_input_type:
        return vol_data.input_strikes
    else: # Need conversion
        if fwd_curve is None:
            raise ValueError(f"Forward curve required for strike conversion but None given: {vol_data.name}")

        # Need to loop over expiries because not all expiries must have the same number of strikes.
        # Therefore we cannot put the strikes into numpy arrays.
        fwds = fwd_curve.value(vol_data.expiries)
        n_times = len(vol_data.expiries)
        if to_type_lw == 'absolute' and vol_data.strike_input_type == 'relative':
            conv_strikes = [vol_data.input_strikes[i] * fwds[i] for i in range(n_times)]
        elif to_type_lw == 'relative' and vol_data.strike_input_type == 'absolute':
            conv_strikes = [vol_data.input_strikes[i] / fwds[i] for i in range(n_times)]
        else:
            raise ValueError(f"Unknown strike type {to_type}: expected absolute or relative")

        return conv_strikes
