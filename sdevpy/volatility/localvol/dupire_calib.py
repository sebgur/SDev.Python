from pathlib import Path
import logging
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sdevpy.maths import constants
from sdevpy.utilities.timegrids import SimpleTimeGridBuilder, BucketTimeGridBuilder
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol, LvMethod
from sdevpy.volatility.localvol.localvol import MatrixLocalVol
log = logging.getLogger(Path(__file__).stem)


def dupire_formula(ivsurf: ImpliedVol, ts: float, te: float, x: npt.ArrayLike) -> npt.ArrayLike:
    """ Calculate Dupire formula on the ImpliedVol, between times ts and te, at moneyness x """
    # In case the LV is analytical, return it immediately
    if ivsurf.lv_method == LvMethod.Analytical:
        return ivsurf.local_vol_step(ts, te, x)
        # return ivsurf.local_vol(ts, x)

    t_threshold = 0.0001
    x_threshold = 0.00001
    iv_threshold = 0.000001

    # Edge case: t = 0
    if ts < t_threshold:
        return ivsurf.black_volatility(te, x, 1.0)

    # Calculate forward variance
    dvar_dt = ivsurf.dvariance_dt(ts, te, x)

    # Distinguish according to method
    match ivsurf.lv_method:
        case LvMethod.ImpliedVol:
            theta, dtheta_dx, d2theta_dx2 = ivsurf.taylor_dx(ts, x)

            # IV = 0
            zero_mask = (theta < iv_threshold)
            pos_theta = np.maximum(theta, iv_threshold)

            xdtheta_dx = x * dtheta_dx
            x2d2theta_dx2 = np.power(x, 2) * d2theta_dx2
            theta_ts = pos_theta * ts
            sqrt_t_d = -np.log(x) / pos_theta + 0.5 * theta_ts
            tmp = np.power(1.0 + sqrt_t_d * xdtheta_dx, 2)
            denominator = theta_ts * (x2d2theta_dx2 - sqrt_t_d * xdtheta_dx * xdtheta_dx) + tmp
        case LvMethod.PDF:
            # sdev = 0
            theta = ivsurf.black_volatility(ts, x, 1.0)
            stdev = theta * np.sqrt(ts)
            stdev_threshold = iv_threshold * np.sqrt(t_threshold)
            zero_mask = (stdev < stdev_threshold)
            pos_stdev = np.maximum(stdev, stdev_threshold)

            dm = -np.log(x) / pos_stdev - 0.5 * pos_stdev
            delta_nm = np.exp(-0.5 * dm * dm) / constants.C_SQRT2PI
            pdf = ivsurf.density(ts, 1.0, x)
            denominator = pos_stdev * x * pdf / delta_nm
        case _:
            raise ValueError(f"Invalid Dupire calculation method: {ivsurf.lv_method}")

    if np.any(zero_mask):
        log.debug(f"Zero-mask found for {np.count_nonzero(zero_mask)} points")

    sigma2 = np.where(zero_mask, 0.0, dvar_dt / denominator)
    sigma2 = np.where(x < x_threshold, dvar_dt, sigma2)

    # print(f"sigma2: {sigma2}")
    neg_mask = (sigma2 < 0.0)
    if np.any(neg_mask):
        log.debug(f"Negative squared vol found for {np.count_nonzero(neg_mask)} points")

    return np.sqrt(np.maximum(sigma2, 0.0))


def calib_lv_dupire(surface: ImpliedVol, **kwargs) -> dict:
    """ Calibrate MatrixLocalVol by Dupire's formula, from a given implied vol surface.
        Passing the time grid is typically what we do when we want to use Black-Scholes
        model but still go through the Dupire calibration, for investigation purposes. """
    # Arguments
    verbose = kwargs.get('verbose', False)
    n_points_per_year = kwargs.get('points_per_year', 10)
    time_buckets = kwargs.get('time_buckets', None)
    n_strikes = kwargs.get('n_strikes', 25)
    lw_percent = kwargs.get('low_percent', 0.01)
    up_percent = kwargs.get('up_percent', 1.0 - lw_percent)
    tmax = kwargs.get('tmax', 2.0)
    t_grid = kwargs.get('t_grid', None)
    grid_in_percents = kwargs.get('grid_in_percents', False) # True to equally space strikes in percents
    conf = None
    if grid_in_percents:
        percents = np.linspace(lw_percent, up_percent, n_strikes)
        conf = norm.ppf(percents)

    # Create time grid
    if t_grid is None:
        base_grid = []
        base_grid.append(tmax)
        base_grid = np.asarray(base_grid)
        if time_buckets is not None:
            t_grid_builder = BucketTimeGridBuilder(include_t0=True, buckets=time_buckets)
        else:
            t_grid_builder = SimpleTimeGridBuilder(include_t0=True, points_per_year=n_points_per_year)
        t_grid_builder.add_grid(base_grid)
        t_grid = t_grid_builder.complete_grid()
    n_times = len(t_grid)

    # Calculate Dupire for suitable dates
    lv = [None] * n_times
    moneynesses = [None] * n_times
    for i in range(0, n_times - 1):
        ts = t_grid[i]
        te = t_grid[i + 1]
        # Create moneynesses axis
        atm_vol = surface.black_volatility(te, 1.0, 1.0) # ATM
        stdev = atm_vol * np.sqrt(te)
        if grid_in_percents:
            m = np.exp(-0.5 * stdev * stdev + stdev * conf)
        else:
            low_k = np.exp(-0.5 * stdev * stdev + stdev * norm.ppf(lw_percent))
            up_k = np.exp(-0.5 * stdev * stdev + stdev * norm.ppf(up_percent))
            m = np.linspace(low_k, up_k, n_strikes)

        moneynesses[i] = m
        lv[i] = dupire_formula(surface, ts, te, m)

        if verbose:
            log.info(f"Iteration {i} from {ts} to {te}")
            log.info(f"Moneynesses: {m}")
            log.info(f"Local vol: {lv[i]}")

    # Set first and last slices to next/previous
    moneynesses[-1], lv[-1] = moneynesses[-2], lv[-2]

    lv_obj = MatrixLocalVol(t_grid, np.log(moneynesses), lv)
    return {'lv': lv_obj, 't_grid': t_grid, 'moneyness': moneynesses, 'lv_matrix': lv}


if __name__ == "__main__":
    print("Hello")
