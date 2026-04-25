import numpy as np
import numpy.typing as npt
from sdevpy.volatility.impliedvol.zerosurface import ZeroSurface, LvMethod
from sdevpy.maths import constants
from sdevpy.utilities.tools import isequal


######### ToDo #############################################
# * Start by using the scalar formula with all float
# * Use it over loops
# * Then try vectorizing it, whereby:
#   - ts of shape(nt,), te of shape (nt,)
#   - x of shape(nt, nx)
#   - return of shape (nt, nx)
# * Make sure that LogMix goes through proper density = PDF


def dupire_formula_single(ivsurf: ZeroSurface, ts: float, te: float, x: float) -> float:
    """ Calculate Dupire formula on the ZeroSurface """
    t_threshold = 0.0001
    x_threshold = 0.00001
    iv_threshold = 0.000001

    # Edge case: t = 0
    if np.abs(ts) < t_threshold:
        return ivsurf.black_volatility(te, x, 1.0)

    # Edge case: moneyness = 0
    dvar_dt = ivsurf.dvariance_dt(ts, te, x)
    print(f"dvar_dt: {dvar_dt}")
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
            print(f"sqrt_t_d: {sqrt_t_d}")
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

    print(f"denominator: {denominator}")

    sigma2 = dvar_dt / denominator
    return (0.0 if sigma2 < 0.0 else np.sqrt(sigma2))


def dupire_formula(ivsurf: ZeroSurface, ts: float, te: float, x: npt.ArrayLike) -> npt.ArrayLike:
    """ Calculate Dupire formula on the ZeroSurface, between times ts and te, at moneyness x """
    t_threshold = 0.0001
    x_threshold = 0.00001
    iv_threshold = 0.000001

    # Edge case: t = 0
    if ts < t_threshold:
        return ivsurf.black_volatility(te, x, 1.0)

    # Calculate forward variance
    dvar_dt = ivsurf.dvariance_dt(ts, te, x)
    print(f"dvar_dt: {dvar_dt}")

    # Edge case: negative forward variance
    # pos_fwd_var_mask = (dvar_dt > 0.0)
    # pos_fwd_var = np.maximum(dvar_dt, 0.0)
    # sqrt_fwd_var = np.sqrt(pos_fwd_var)

    # Edge case: moneyness = 0
    # small_m_mask = (x < x_threshold)
        # return np.sqrt(dvar_dt)

    # Distinguish according to method
    match ivsurf.lv_method:
        case LvMethod.ImpliedVol:
            # theta = ivsurf.black_volatility(ts, x, 1.0)
            theta, dtheta_dx, d2theta_dx2 = ivsurf.taylor_dx(ts, x)

            # IV = 0
            zero_mask = (theta < iv_threshold)
            pos_theta = np.maximum(theta, iv_threshold)
            # numerator = np.where(neg_iv_mask, 0.0, sqrt_fwd_var)

            xdtheta_dx = x * dtheta_dx
            x2d2theta_dx2 = np.power(x, 2) * d2theta_dx2
            theta_ts = pos_theta * ts
            # xdtheta_dx = ivsurf.dvolatility_dx(ts, x)
            # x2d2theta_dx2 = ivsurf.d2volatility_dx2(ts, x)
            # xdtheta_dx *= x
            # x2d2theta_dx2 *= x * x
            sqrt_t_d = -np.log(x) / pos_theta + 0.5 * theta_ts
            print(f"sqrt_t_d: {sqrt_t_d}")
            tmp = np.power(1.0 + sqrt_t_d * xdtheta_dx, 2)
            # tmp *= tmp
            denominator = theta_ts * (x2d2theta_dx2 - sqrt_t_d * xdtheta_dx * xdtheta_dx) + tmp
        case LvMethod.PDF:
            # sdev = 0
            theta = ivsurf.black_volatility(ts, x, 1.0)
            stdev = theta * np.sqrt(ts)
            stdev_threshold = iv_threshold * np.sqrt(t_threshold)
            zero_mask = (stdev < stdev_threshold)
            pos_stdev = np.maximum(stdev, stdev_threshold)
            # numerator = np.where(neg_stdev_mask, 0.0, sqrt_fwd_var)

            dm = -np.log(x) / pos_stdev - 0.5 * pos_stdev
            delta_nm = np.exp(-0.5 * dm * dm) / constants.C_SQRT2PI
            pdf = ivsurf.density(ts, 1.0, x)
            denominator = pos_stdev * x * pdf / delta_nm
        case _:
            raise ValueError(f"Invalid Dupire calculation method: {ivsurf.lv_method}")

    # print(f"dvar_dt: {dvar_dt}")
    # print(f"denominator: {denominator}")

    # sigma2 = dvar_dt / denominator
    print(f"denominator: {denominator}")
    sigma2 = np.where(zero_mask, 0.0, dvar_dt / denominator)
    # sigma2 = dvar_dt / denominator
    # return (0.0 if sigma2 < 0.0 else np.sqrt(sigma2))
    sigma2 = np.where(x < x_threshold, dvar_dt, sigma2)
    return np.sqrt(np.maximum(sigma2, 0.0))


if __name__ == "__main__":
    from sdevpy.volatility.impliedvol.models.logmix import LogMix
    from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
    from sdevpy.volatility.impliedvol.models.tssvi2 import TsSvi2

    t_grid = np.asarray([0.25, 1.0, 2.0])
    ts = t_grid[:-1]
    te = t_grid[1:]
    x = [[0.99, 1.0, 1.01], [0.9, 0.95, 1.0, 1.05, 1.1]]#, [0.8, 1.0, 1.2]]
    # ts, te = 0.5, 1.0
    # x = 1.2

    # Set IV surface models
    surface1 = LogMix(3)
    surface1.update_params(surface1.initial_point())

    surface2 = TsSvi1()
    surface2.update_params(surface2.initial_point())

    surface3 = TsSvi2()
    surface3.update_params(surface3.initial_point())

    lv1, lv2, lv3 = [], [], []
    for i in range(1):#len(ts)):
        t1 = ts[i]
        t2 = te[i]
        m = x[i]
        print(f"Iteration {i+1} from {t1} to {t2}")
        print(f"Moneynesses: {m}")
        for m_ in m:
            # lv1_ = dupire_formula_single(surface1, t1, t2, m_)
            # lv1.append(lv1_)

            lv2_ = dupire_formula_single(surface2, t1, t2, m_)
            lv2.append(lv2_)

            lv3_ = dupire_formula_single(surface3, t1, t2, m_)
            lv3.append(lv3_)

    # print(f"Model 1: {lv1}")
    print(f"Model 2: {lv2}")
    print(f"Model 3: {lv3}")

    # Vectorize
    lv1_vec, lv2_vec, lv3_vec = [], [], []
    for i in range(1):#len(ts)):
        t1 = ts[i]
        t2 = te[i]
        m = np.asarray(x[i])
        print(f"Iteration {i+1} from {t1} to {t2}")
        print(f"Moneynesses: {m}")
        # lv1_ = dupire_formula(surface1, t1, t2, m)
        # lv1_vec.append(lv1_)

        lv2_ = dupire_formula(surface2, t1, t2, m)
        lv2_vec.append(lv2_)

        lv3_ = dupire_formula(surface3, t1, t2, m)
        lv3_vec.append(lv3_)

    print(f"Model 2: {lv2_vec}")
    print(f"Model 3: {lv3_vec}")

    print(lv2 - np.asarray(lv2_vec))
    print(lv3 - np.asarray(lv3_vec))

    # lv12 = dupire_formula(surface1, ts, te, x)
    # print(lv12)
