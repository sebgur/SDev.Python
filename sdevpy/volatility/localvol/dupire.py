from pathlib import Path
import logging
import datetime as dt
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sdevpy.maths import constants
from sdevpy.utilities.tools import isequal
from sdevpy.utilities.timegrids import SimpleTimeGridBuilder
from sdevpy.volatility.impliedvol.impliedvol import ImpliedVol, LvMethod
from sdevpy.market import eqvolsurface as vsurf
log = logging.getLogger(Path(__file__).stem)


######### ToDo #############################################
# * Implement calib_lv_dupire


def dupire_formula_single(ivsurf: ImpliedVol, ts: float, te: float, x: float) -> float:
    """ ToDo: remove when all done """
    t_threshold = 0.0001
    x_threshold = 0.00001
    iv_threshold = 0.000001

    # Edge case: t = 0
    if np.abs(ts) < t_threshold:
        return ivsurf.black_volatility(te, x, 1.0)

    # Edge case: moneyness = 0
    dvar_dt = ivsurf.dvariance_dt(ts, te, x)
    # print(f'dvar_dt: {dvar_dt}')
    if x < x_threshold:
        return np.sqrt(dvar_dt)

    # Distinguish according to method
    theta = ivsurf.black_volatility(ts, x, 1.0)
    match ivsurf.lv_method:
        case LvMethod.ImpliedVol:
            # IV = 0
            if isequal(theta, 0.0, iv_threshold):
                return 0.0

            xdtheta_dx = ivsurf.dvolatility_dx(ts, x)
            x2d2theta_dx2 = ivsurf.d2volatility_dx2(ts, x)
            xdtheta_dx *= x
            x2d2theta_dx2 *= x * x
            sqrt_t_d = -np.log(x) / theta + 0.5 * theta * ts
            tmp = 1.0 + sqrt_t_d * xdtheta_dx
            tmp *= tmp
            denominator = theta * ts * (x2d2theta_dx2 - sqrt_t_d * xdtheta_dx * xdtheta_dx) + tmp
        case LvMethod.PDF:
            # sdev = 0
            stddev = theta * np.sqrt(ts)
            if isequal(stddev, 0.0, iv_threshold * np.sqrt(t_threshold)):
                return 0.0

            dm = -np.log(x) / stddev - 0.5 * stddev
            delta_nm = np.exp(-0.5 * dm * dm) / constants.C_SQRT2PI
            pdf = ivsurf.density(ts, 1.0, x)
            denominator = stddev * x * pdf / delta_nm
        case _:
            raise ValueError(f"Invalid Dupire calculation method: {ivsurf.lv_method}")

    sigma2 = dvar_dt / denominator
    return (0.0 if sigma2 < 0.0 else np.sqrt(sigma2))


def dupire_formula(ivsurf: ImpliedVol, ts: float, te: float, x: npt.ArrayLike) -> npt.ArrayLike:
    """ Calculate Dupire formula on the ImpliedVol, between times ts and te, at moneyness x """
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


def calib_lv_dupire(surface: ImpliedVol, valdate: dt.datetime, config: dict, **kwargs) -> dict:
    """ Calibrate MatrixLocalVol by Dupire's formula, from a given implied vol
        surface. """
    # Arguments
    verbose = kwargs.get('verbose', False)
    n_points_per_year = kwargs.get('points_per_year', 10)
    n_strikes = kwargs.get('n_strikes', 25)
    lw_percent = kwargs.get('low_percent', 0.01)
    up_percent = kwargs.get('up_percent', 1.0 - lw_percent)

    # Create time grid starting at 1D
    t_grid_builder = SimpleTimeGridBuilder(include_t0=True, points_per_year=n_points_per_year)
    short_end_grid = np.asarray([1.0 / 365, 7.0 / 365, 14.0 / 365, 30.0 / 365, 1.0, 2.0])
    t_grid_builder.add_grid(short_end_grid)
    t_grid = t_grid_builder.get_grid()
    t_grid = t_grid_builder.complete_grid()
    n_times = len(t_grid)

    # Calculate Dupire for suitable dates
    lv = [[None]] * n_times
    moneynesses = [[None]] * n_times
    for i in range(1, n_times - 1):
        ts = t_grid[i]
        te = t_grid[i + 1]
        # Create moneynesses axis
        atm_vol = surface.black_volatility(ts, 1.0, 1.0) # ATM
        stdev = atm_vol * np.sqrt(ts)
        low_k = np.exp(-0.5 * stdev * stdev + stdev * norm.ppf(lw_percent))
        up_k = np.exp(-0.5 * stdev * stdev + stdev * norm.ppf(up_percent))
        m = np.linspace(low_k, up_k, n_strikes)

        moneynesses[i] = m
        lv[i] = dupire_formula(surface, ts, te, m)

        if verbose:
            log.info(f"Iteration {i+1} from {ts} to {te}")
            log.info(f"Moneynesses: {m}")
            log.info(f"Local vol: {lv[i]}")

    # Set first and last slices to next/previous
    moneynesses[0], lv[0] = moneynesses[1], lv[1]
    moneynesses[-1], lv[-1] = moneynesses[-2], lv[-2]

    return {'t_grid': t_grid, 'moneyness': moneynesses, 'lv': lv}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sdevpy.volatility.impliedvol.models.logmix import LogMix
    from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
    from sdevpy.volatility.impliedvol.models.tssvi2 import TsSvi2

    t_grid = np.asarray([0.25, 1.0, 2.0])
    ts = t_grid[:-1]
    te = t_grid[1:]
    x = [[0.99, 1.0, 1.01], [0.9, 0.95, 1.0, 1.05, 1.1]]#, [0.8, 1.0, 1.2]]

    # Set IV surface models
    surface1 = LogMix(3)
    surface1.update_params(surface1.initial_point())

    surface2 = TsSvi1()
    surface2.update_params(surface2.initial_point())

    surface3 = TsSvi2()
    surface3.update_params(surface3.initial_point())

    # Single
    print("<><><><> Single <><><><>")
    lv1_sin, lv2_sin, lv3_sin = [], [], []
    for i in range(len(ts)):
        t1 = ts[i]
        t2 = te[i]
        m = x[i]
        print(f"Iteration {i+1} from {t1} to {t2}")
        print(f"Moneynesses: {m}")
        lv1, lv2, lv3, lv4 = [], [], [], []
        for m_ in m:
            # lv1_ = dupire_formula_single(surface1, t1, t2, m_)
            # lv1.append(lv1_)

            lv2_ = dupire_formula_single(surface2, t1, t2, m_)
            lv2.append(lv2_)

            # lv3_ = dupire_formula_single(surface3, t1, t2, m_)
            # lv3.append(lv3_)

        lv1_sin.append(lv1)
        lv2_sin.append(lv2)
        lv3_sin.append(lv3)

    # Vectorize
    print("<><><><> Vectorize <><><><>")
    lv1_vec, lv2_vec, lv3_vec = [], [], []
    for i in range(len(ts)):
        t1 = ts[i]
        t2 = te[i]
        m = np.asarray(x[i])
        print(f"Iteration {i+1} from {t1} to {t2}")
        print(f"Moneynesses: {m}")

        # lv1_ = dupire_formula(surface1, t1, t2, m)
        # lv1_vec.append(lv1_)

        lv2_ = dupire_formula(surface2, t1, t2, m)
        lv2_vec.append(lv2_)

        # lv3_ = dupire_formula(surface3, t1, t2, m)
        # lv3_vec.append(lv3_)

    print("<><><><> Compare <><><><>")
    for i in range(len(ts)):
        print(f"Iteration {i+1}")
        # for j in range(len(lv1_sin[i])):
        #     print(f"LV1(sin): {lv1_sin[i][j]}")
        #     print(f"LV1(vec): {lv1_vec[i][j]}")

        for j in range(len(lv2_sin[i])):
            print(f"LV2(sin): {lv2_sin[i][j]}")
            print(f"LV2(vec): {lv2_vec[i][j]}")

        # for j in range(len(lv3_sin[i])):
        #     print(f"LV3(sin): {lv3_sin[i][j]}")
        #     print(f"LV3(vec): {lv3_vec[i][j]}")

    # Calibrate
    result = calib_lv_dupire(surface3, None, None, points_per_year=4, verbose=True)
    lv = result['lv']
    m = result['moneyness']
    t = result['t_grid']

    # Display LV
    idx = 6
    # print(lv)

    # Time plot
    lvs = [lv[i][idx] for i in range(len(lv))]
    plt.plot(t, lvs)
    plt.show()

    # Strike plot
    ms = m[idx]
    lvs = lv[idx]
    plt.plot(ms, lvs)
    plt.show()
