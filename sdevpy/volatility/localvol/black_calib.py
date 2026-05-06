from pathlib import Path
import logging
import numpy as np
import numpy.typing as npt
import datetime as dt
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol
from sdevpy.volatility.localvol.localvol import VectorLocalVol, ConstantLocalVol
from sdevpy.utilities import timegrids
log = logging.getLogger(Path(__file__).stem)


def calib_lv_black(surface: ImpliedVol, dates: list[dt.datime], strikes: list[float],
                   fwds: list[float]) -> dict:
    """ Calibrate VectorLocalVol or ConstantLocalVol by Black closed-form.
        Pick maturities and strikes to calibrated to. """
    # Arguments
    # verbose = kwargs.get('verbose', False)

    # Check sizes
    n_times = len(dates)
    if len(strikes) != n_times:
        raise ValueError("Inconsistent sizes between expiries and strikes")

    if len(fwds) != n_times:
        raise ValueError("Inconsistent sizes between expiries and forwards")

    # Calibrate
    valdate = surface.base_date
    times = timegrids.model_time(valdate, dates)
    ivols = surface.black_volatility(times, strikes, fwds)
    lv = None
    if n_times < 1:
        raise ValueError("No option dates specified")
    elif n_times == 1:
        lv = ConstantLocalVol(ivols[0])
    else:
        lvols = []
        for i in range(len(times)):
            if i == 0:
                lvols.append(ivols[0])
            else:
                ts, vols = times[i - 1], ivols[i - 1]
                te, vole = times[i], ivols[i]
                lvols.append(np.sqrt((vole**2 * te - vols**2 * ts) / (te - ts)))

        lv = VectorLocalVol(times, lvols)

    return {'lv': lv}


if __name__ == "__main__":
    print("Hello")
