""" Prototype of a branch-skipping strike_from_delta, for benchmarking only.
    Mirrors fx_deltastrike.strike_from_delta exactly, except:
      - each of the three branches is computed only when some element actually needs it
      - convergence checks use ndarray.max() instead of np.nanmax()
"""
import numpy as np
from scipy.special import ndtr, ndtri
from sdevpy.volatility.fx.fx_deltastrike import (
    _arr, _phi_from_option_type, _d1d2, bs_delta, fx_market_yearfraction, StrikeSolution)


def _bisect(func, target, x_lo, x_hi, increasing, tol=1e-12, max_iter=100):
    lo, hi = x_lo.copy(), x_hi.copy()
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore"):
            val = func(np.exp(mid)) - target
        go_right = val < 0 if increasing else val > 0
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
        if (hi - lo).max() < tol:      # was np.nanmax
            break
    return np.exp(0.5 * (lo + hi))


def _ternary_max(func, x_lo, x_hi, iters=100):
    lo, hi = x_lo.copy(), x_hi.copy()
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        with np.errstate(over="ignore"):
            f1, f2 = func(np.exp(m1)), func(np.exp(m2))
        go_right = f1 < f2
        lo = np.where(go_right, m1, lo)
        hi = np.where(go_right, hi, m2)
        if (hi - lo).max() < 1e-12:    # was np.nanmax
            break
    x_star = 0.5 * (lo + hi)
    k_star = np.exp(x_star)
    return k_star, func(k_star)


def strike_from_delta(valdate, expiry, spot, df_f, df_d, sigma, delta, option_type,
                      prem_adjusted, spot_delta, double_root_preference="small",
                      bracket_width_sigma_mult=15.0, bracket_width_floor=8.0,
                      tol_existence=1e-9, tol_residual=1e-6, max_iter=100):
    t = fx_market_yearfraction(valdate, expiry)
    s, df_f, df_d, t, sigma, delta = np.broadcast_arrays(
        *[_arr(a) for a in (spot, df_f, df_d, t, sigma, delta)])
    phi = np.broadcast_to(_phi_from_option_type(option_type), s.shape).astype(float)
    prem_adjusted = np.broadcast_to(_arr(prem_adjusted).astype(bool), s.shape)

    if np.any(sigma <= 0) or np.any(t <= 0):
        raise ValueError("sigma and T must be strictly positive everywhere")

    f = s * df_f / df_d
    use_spot_delta = np.broadcast_to(_arr(spot_delta).astype(bool), s.shape)
    disc = np.where(use_spot_delta, df_f, 1.0)

    is_pa = prem_adjusted
    is_put = phi < 0
    is_call = ~is_put
    need_cf = bool(np.any(~is_pa))
    need_pa_put = bool(np.any(is_pa & is_put))
    need_pa_call = bool(np.any(is_pa & is_call))

    sqrt_t = np.sqrt(t)
    b = bracket_width_sigma_mult * sigma * sqrt_t + bracket_width_floor

    nan_like = np.full(s.shape, np.nan)

    # Branch 1: closed form
    if need_cf:
        x_cf = phi * delta / disc
        with np.errstate(invalid="ignore"):
            d1_cf = phi * ndtri(x_cf)
        k_cf = f * np.exp(-d1_cf * sigma * sqrt_t + 0.5 * sigma ** 2 * t)
        valid_cf = (x_cf > 0) & (x_cf < 1)
    else:
        k_cf, valid_cf = nan_like, np.zeros(s.shape, dtype=bool)

    # Branch 2: PA put
    if need_pa_put:
        def _put_pa_delta(k):
            _, d2 = _d1d2(f, k, sigma, t)
            return -disc * (k / f) * ndtr(-d2)

        x_lo_put, x_hi_put = np.log(f) - b, np.log(f) + b
        for _ in range(40):
            val_lo = _put_pa_delta(np.exp(x_lo_put)) - delta
            val_hi = _put_pa_delta(np.exp(x_hi_put)) - delta
            need_lo, need_hi = val_lo < 0, val_hi > 0
            if not (np.any(need_lo) or np.any(need_hi)):
                break
            width = x_hi_put - x_lo_put
            x_lo_put = np.where(need_lo, x_lo_put - width, x_lo_put)
            x_hi_put = np.where(need_hi, x_hi_put + width, x_hi_put)
        k_put_pa = _bisect(_put_pa_delta, delta, x_lo_put, x_hi_put, False, max_iter=max_iter)
    else:
        k_put_pa = nan_like

    # Branch 3: PA call
    if need_pa_call:
        def _call_pa_delta(k):
            _, d2 = _d1d2(f, k, sigma, t)
            return disc * (k / f) * ndtr(d2)

        x_lo_call, x_hi_call = np.log(f) - b, np.log(f) + b
        k_max, delta_max = _ternary_max(_call_pa_delta, x_lo_call, x_hi_call, iters=max_iter)
        x_k_max = np.log(k_max)
        diff = delta - delta_max
        no_sol_call = diff > tol_existence
        one_sol_call = np.abs(diff) <= tol_existence
        k_call_left = _bisect(_call_pa_delta, delta, x_lo_call, x_k_max, True, max_iter=max_iter)
        k_call_right = _bisect(_call_pa_delta, delta, x_k_max, x_hi_call, False, max_iter=max_iter)
        if double_root_preference not in ("small", "large"):
            raise ValueError("double_root_preference must be 'small' or 'large'.")
        k_call_primary = k_call_left if double_root_preference == "small" else k_call_right
        k_call_secondary = k_call_right if double_root_preference == "small" else k_call_left
        k_call_primary = np.where(one_sol_call, k_max, k_call_primary)
        k_call_secondary = np.where(one_sol_call, np.nan, k_call_secondary)
        k_call_primary = np.where(no_sol_call, np.nan, k_call_primary)
        k_call_secondary = np.where(no_sol_call, np.nan, k_call_secondary)
        n_sol_call = np.where(no_sol_call, 0, np.where(one_sol_call, 1, 2))
    else:
        k_call_primary = k_call_secondary = nan_like
        delta_max = nan_like
        n_sol_call = np.zeros(s.shape, dtype=int)

    k = np.where(~is_pa, k_cf, np.where(is_put, k_put_pa, k_call_primary))
    k_alt = np.where(~is_pa, np.nan, np.where(is_put, np.nan, k_call_secondary))
    n_solutions = np.where(~is_pa, np.where(valid_cf, 1, 0),
                           np.where(is_put, 1, n_sol_call)).astype(int)
    delta_max_abs = np.where(is_pa & is_call, delta_max, np.nan)

    with np.errstate(invalid="ignore"):
        residual = bs_delta(f, k, sigma, t, phi, disc, prem_adjusted) - delta
    valid = np.where(~is_pa, valid_cf, True)
    valid = valid & (np.abs(residual) < tol_residual) & ~np.isnan(k)
    n_solutions = np.where(~valid & (n_solutions > 0), 0, n_solutions)
    k = np.where(valid, k, np.nan)

    return StrikeSolution(k=k, k_alt=k_alt, n_solutions=n_solutions, delta_max_abs=delta_max_abs,
                          residual=residual, valid=valid, used_spot_delta=use_spot_delta,
                          prem_adjusted=prem_adjusted)
