import numpy as np
import datetime as dt
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol
from sdevpy.volatility.localvol.localvol import VectorLocalVol, ConstantLocalVol
from sdevpy.utilities import timegrids


def calib_lv_black(surface: ImpliedVol, dates: list[dt.datetime], strikes: list[float],
                   fwds: list[float]) -> dict:
    """ Calibrate VectorLocalVol or ConstantLocalVol by Black closed-form, from given IV surface.
        Pick maturities and strikes to calibrated to. """
    strikes = np.asarray(strikes)
    fwds = np.asarray(fwds)

    # Check sizes
    n_times = len(dates)
    if len(strikes) != n_times:
        raise ValueError("Inconsistent sizes between expiries and strikes")

    if len(fwds) != n_times:
        raise ValueError("Inconsistent sizes between expiries and forwards")

    # Calculate target vols
    valdate = surface.base_date
    times = timegrids.model_time(valdate, dates)
    ivols = surface.black_volatility(times, strikes, fwds)

    # Calibrate
    result = calib_black_from_vols(times, ivols)
    return result


def calib_black_from_vols(expiries: list[float], ivols: list[float]) -> dict:
    """ Calibrate VectorLocalVol or ConstantLocalVol by Black closed-form, from given expiry times
        and implied vols """
    # Check sizes
    n_times = len(expiries)
    if len(ivols) != n_times:
        raise ValueError("Inconsistent sizes between expiries and implied vols")

    # Calibrate
    lv = None
    if n_times < 1:
        raise ValueError("No option dates specified")
    elif n_times == 1:
        lv = ConstantLocalVol(ivols[0])
    else:
        lvols = []
        for i in range(len(expiries)):
            if i == 0:
                lvols.append(ivols[0])
            else:
                ts, vols = expiries[i - 1], ivols[i - 1]
                te, vole = expiries[i], ivols[i]
                lvols.append(np.sqrt((vole**2 * te - vols**2 * ts) / (te - ts)))

        lv = VectorLocalVol(expiries, lvols)

    return {'lv': lv}


if __name__ == "__main__":
    print("Hello")
