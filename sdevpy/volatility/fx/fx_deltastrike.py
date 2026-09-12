"""
Convert FX-option delta quotes (e.g. "25-delta put", "10-delta call") into Garman-Kohlhagen strikes, fully vectorized
over an arbitrary broadcastable shape (many deltas x many maturities x many smiles at once).

Why this isn't a one-liner
---------------------------
delta = phi * disc * N(phi*d1)          <- "plain" delta:      closed form
delta = phi * disc * (K/F) * N(phi*d2)  <- "premium-adjusted": NOT closed form in K

Plain (non premium-adjusted) delta is a strictly monotonic function of K, so K(delta) has a closed-form solution
via the inverse normal CDF.

Premium-adjusted delta is *not* monotonic in K for calls: as a function of K it rises from 0, reaches a maximum at
some K_max, then falls back to 0. Consequently, for a premium-adjusted call:
    - if |target delta| > max achievable delta  -> NO strike solves it
    - if |target delta| == max achievable delta -> exactly ONE strike (K_max)
    - if |target delta| <  max achievable delta -> exactly TWO strikes solve it
For premium-adjusted puts the function is monotonic (unbounded), so there is always exactly one solution, but it still
requires numerical inversion since there is no closed form.

This module handles both cases, picks a market-convention solution when there are two roots (the smaller strike,
per Reiswich & Wystup, "FX Volatility Smile Construction"), and flags every element with a validity/diagnostic status
rather than silently returning nonsense.

It also switches between "spot delta" (includes the foreign-currency discount factor) and "forward delta" (excludes it)
conventions automatically at a configurable maturity cutoff (1Y by default), which is the standard market practice for
long-dated FX options.

Everything below is implemented with numpy array operations only -- the iterative solvers (bisection for monotonic
branches, ternary search for the unimodal call/premium-adjusted case) advance *all* elements of the input arrays
simultaneously per iteration, so a whole delta/maturity grid is solved in the same number of iterations as a single
quote. No Python-level loop over individual quotes, and no per-element calls into scipy.optimize.
"""
from dataclasses import dataclass
import datetime as dt
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
# from sdevpy.volatility.fx.fx_strangle import fx_market_yearfraction


def fx_market_yearfraction(valdate: dt.datetime, expiry: dt.datetime) -> float:
    """ Yearfraction to put into Black-Scholes formula for the standard deviation that gets root-squared
        and multiplied by the implied vols for the pricing of options.
        WARNING: this is not meant to be used anywhere else. For instance the calculation of rates and/or
        discount factors have no reasons to follow this same convention.
        TODO: for now we use the basic model convention. Based on our information, this should be switched
        to Act/365 Fixed. """
    return timegrids.model_time(valdate, expiry)


def is_spot_delta_tenor(tenor_str: str, cutoff: str = '1Y') -> bool:
    return dts.tenor_leq(tenor_str, cutoff)


def _arr(x) -> npt.ArrayLike:
    return np.asarray(x, dtype=float)


def _phi_from_option_type(option_type) -> npt.ArrayLike:
    """ Map 'C'/'P' (any case, numpy array of strings/objects) or +-1 to phi = +-1 """
    ot = np.asarray(option_type)
    if ot.dtype.kind in ("U", "S", "O"):
        upper = np.array([str(v).strip().upper()[0] for v in ot.ravel()]).reshape(ot.shape)
        phi = np.where(upper == "C", 1.0, np.where(upper == "P", -1.0, np.nan))
        if np.isnan(phi).any():
            raise ValueError("option_type entries must be 'C'/'P' (or +1/-1).")
        return phi
    else:
        phi = np.sign(_arr(option_type))
        phi = np.where(phi == 0, 1.0, phi)
        return phi


def _d1d2(f: npt.ArrayLike, k: npt.ArrayLike, sigma: npt.ArrayLike, t: npt.ArrayLike):
    # K can legitimately hit the extreme edges of a search bracket (~exp(+-huge)), which can underflow/overflow
    # in float64: clip to keep log()/division finite. This never affects the *reported* solution, it only keeps
    # the solver's intermediate probing well-defined so we get a clean "no root here" signal instead of a numpy
    # warning.
    k = np.clip(k, 1e-300, 1e300)
    sqrt_t = np.sqrt(t)
    vol_sqrt_t = sigma * sqrt_t
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(f / k) + 0.5 * sigma ** 2 * t) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_delta(f, k, sigma, t, phi, disc, prem_adjusted) -> npt.ArrayLike:
    """ Vectorized Garman-Kohlhagen delta.
        disc: multiplicative factor distinguishing spot delta (disc = exp(-r_f*T)) from forward delta (disc = 1). """
    f, k, sigma, t, phi, disc = np.broadcast_arrays(*[_arr(a) for a in (f, k, sigma, t, phi, disc)])
    prem_adjusted = np.broadcast_to(_arr(prem_adjusted).astype(bool), f.shape)
    d1, d2 = _d1d2(f, k, sigma, t)
    raw = phi * norm.cdf(phi * d1)
    pa = phi * (k / f) * norm.cdf(phi * d2)
    return np.where(prem_adjusted, pa, raw) * disc


def _vectorized_bisect(func, target, x_lo, x_hi, increasing: bool,
                        tol: float = 1e-12, max_iter: int = 100) -> npt.ArrayLike:
    """
    Bisection in log(K) space (guarantees K > 0, scale-invariant across currency pairs
    with very different spot magnitudes, e.g. JPY crosses vs EURUSD).

    `func` must be monotonic (increasing or decreasing, as declared) in K on
    [exp(x_lo), exp(x_hi)], and the bracket must actually contain the root
    (callers are responsible for supplying/expanding valid brackets).
    """
    lo = x_lo.copy()
    hi = x_hi.copy()
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        with np.errstate(over="ignore"):
            val = func(np.exp(mid)) - target
        if increasing:
            go_right = val < 0
        else:
            go_right = val > 0
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
        # cheap, safe early exit
        if np.nanmax(hi - lo) < tol:
            break
    return np.exp(0.5 * (lo + hi))


def _vectorized_ternary_max(func, x_lo, x_hi, iters: int = 100):
    """
    Ternary search for the maximum of a (provably) unimodal function of K over
    [exp(x_lo), exp(x_hi)], in log(K) space. Returns (K_argmax, func(K_argmax))
    """
    lo = x_lo.copy()
    hi = x_hi.copy()
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        with np.errstate(over="ignore"):
            f1 = func(np.exp(m1))
            f2 = func(np.exp(m2))
        go_right = f1 < f2  # max is to the right of m1
        lo = np.where(go_right, m1, lo)
        hi = np.where(go_right, hi, m2)
        if np.nanmax(hi - lo) < 1e-12:
            break
    x_star = 0.5 * (lo + hi)
    k_star = np.exp(x_star)
    return k_star, func(k_star)


# Result container
@dataclass
class StrikeSolution:
    k: npt.ArrayLike               # recommended strike (NaN where no valid solution)
    k_alt: npt.ArrayLike           # the *other* root when two exist, else NaN
    n_solutions: npt.ArrayLike     # 0, 1, or 2 (int array)
    delta_max_abs: npt.ArrayLike   # max achievable |premium-adjusted call delta|; NaN elsewhere
    residual: npt.ArrayLike        # bs_delta(K) - target_delta, recomputed for the *chosen* K
    valid: npt.ArrayLike           # bool: True iff `residual` is within tolerance (i.e. K is trustworthy)
    used_spot_delta: npt.ArrayLike # bool: True where the spot-delta convention was used (vs forward-delta)
    prem_adjusted: npt.ArrayLike   # bool: echoed back, per element

    def __repr__(self):
        return (f"StrikeSolution(K={self.k}, n_solutions={self.n_solutions}, valid={self.valid})")


# Main entry point
def strike_from_delta(valdate: dt.datetime, expiry: npt.ArrayLike, spot: npt.ArrayLike,
                      df_f: npt.ArrayLike, df_d: npt.ArrayLike,
                      sigma: npt.ArrayLike, delta: npt.ArrayLike, option_type: npt.ArrayLike,
                      prem_adjusted: npt.ArrayLike=False, spot_delta: npt.ArrayLike=None,
                      spot_delta_cutoff: float=1.0, double_root_preference: str="small",
                      bracket_width_sigma_mult: float=15.0, bracket_width_floor: float=8.0,
                      tol_existence: float=1e-9, tol_residual: float=1e-6, max_iter: int=100) -> StrikeSolution:
    """
    Invert Garman-Kohlhagen delta quotes into strikes. Fully vectorized: all array arguments are broadcast
    together, so you can pass e.g. sigma/delta/option_type as a (n_maturities, n_deltas) grid
    and T/df_f/df_d as (n_maturities, 1), and get a (n_maturities, n_deltas) grid of strikes back in one call.

    Parameters
    ----------
    spot: spot FX rate S (domestic per foreign unit)
    d_f, d_d: foreign/domestic discount factors
    sigma: Black-Scholes volatility
    delta: signed target delta (e.g. +0.25 for a 25-delta call, -0.25 for a 25-delta put), expressed in
           whichever convention (spot vs forward, premium-adjusted or not) is implied by the other flags below
           option_type : 'C'/'P' or +1/-1, broadcastable with the other inputs
    prem_adjusted: bool or bool array whether each quote uses premium-adjusted delta. Currency-pair/market
                   dependent. Pass it explicitly per quote or as a single bool for all quotes.
    spot_delta: optional bool/bool-array override. If None (default), the convention is chosen automatically
                per element: spot delta for T <= spot_delta_cutoff, forward delta for T > spot_delta_cutoff. This
                is the standard market rule (spot delta becomes a poor hedge-ratio proxy for long-dated options
                once forward points dominate).
    spot_delta_cutoff: maturity (in years) at which the convention switches (1.0 = 1Y, the market standard, tooverride
                       if needed for a specific pair).
    double_root_preference: 'small' (default, market convention) or 'large', which of the two roots to report as `K`
                            when a premium-adjusted call has two solutions. Both roots are always available via `K_alt`.
    bracket_width_sigma_mult, bracket_width_floor: control how wide (in units of log-strike) the numerical search
                                                   brackets are. The defaults are generous (many sigma*sqrt(T) wide)
                                                   and should not need changing.
    tol_existence: tolerance (in delta units) used to classify "no solution" vs "exactly one" vs "two solutions"
                   for premium-adjusted calls, relative to the achievable maximum delta.
    tol_residual: after solving, the delta implied by the returned strike is recomputed and compared to the target;
                  if the discrepancy exceeds this, the element is marked invalid regardless of how it was classified.
    max_iter: iterations for the bisection/ternary solvers.

    --------
    Returns: StrikeSolution container
    """
    ####
    t = fx_market_yearfraction(valdate, expiry)
    ####

    # Broadcast everything
    s, df_f, df_d, t, sigma, delta = np.broadcast_arrays(*[_arr(a) for a in (spot, df_f, df_d, t, sigma, delta)])
    phi = np.broadcast_to(_phi_from_option_type(option_type), s.shape).astype(float)
    prem_adjusted = np.broadcast_to(_arr(prem_adjusted).astype(bool), s.shape)

    if np.any(sigma <= 0) or np.any(t <= 0):
        raise ValueError("sigma and T must be strictly positive everywhere.")

    f = s * df_f / df_d

    if spot_delta is None:
        use_spot_delta = t <= spot_delta_cutoff
    else:
        use_spot_delta = np.broadcast_to(_arr(spot_delta).astype(bool), s.shape)
    disc = np.where(use_spot_delta, df_f, 1.0)

    # shape = s.shape
    sqrt_t = np.sqrt(t)
    b = bracket_width_sigma_mult * sigma * sqrt_t + bracket_width_floor  # log-K half-width

    # Branch 1: plain (non premium-adjusted) delta -> closed form
    #   delta = phi * disc * N(phi*d1)   =>   d1 = phi * N^{-1}(phi*delta/disc)
    x_cf = phi * delta / disc
    with np.errstate(invalid="ignore"):
        d1_cf = phi * norm.ppf(x_cf)  # NaN automatically outside (0,1) -- that's correct: no solution
    k_cf = f * np.exp(-d1_cf * sigma * sqrt_t + 0.5 * sigma ** 2 *t)
    valid_cf = (x_cf > 0) & (x_cf < 1)

    # Branch 2: premium-adjusted PUT delta -> monotonic, unique root
    def _put_pa_delta(k):
        _, d2 = _d1d2(f, k, sigma, t)
        return -disc * (k / f) * norm.cdf(-d2)

    x_lo_put = np.log(f) - b
    x_hi_put = np.log(f) + b
    # make sure the bracket actually straddles the (monotonically decreasing) root;
    # expand geometrically on whichever side is needed (defensive -- normally the
    # generous default B already suffices)
    for _ in range(40):
        val_lo = _put_pa_delta(np.exp(x_lo_put)) - delta
        val_hi = _put_pa_delta(np.exp(x_hi_put)) - delta
        need_lo = val_lo < 0          # want delta(lo) > target ; if not, push lo further left
        need_hi = val_hi > 0          # want delta(hi) < target ; if not, push hi further right
        if not (np.any(need_lo) or np.any(need_hi)):
            break
        width = x_hi_put - x_lo_put
        x_lo_put = np.where(need_lo, x_lo_put - width, x_lo_put)
        x_hi_put = np.where(need_hi, x_hi_put + width, x_hi_put)

    k_put_pa = _vectorized_bisect(_put_pa_delta, delta, x_lo_put, x_hi_put, increasing=False, max_iter=max_iter)

    # Branch 3: premium-adjusted CALL delta -> unimodal (hump), 0/1/2 roots
    def _call_pa_delta(k):
        _, d2 = _d1d2(f, k, sigma, t)
        return disc * (k / f) * norm.cdf(d2)

    x_lo_call = np.log(f) - b
    x_hi_call = np.log(f) + b
    k_max, delta_max = _vectorized_ternary_max(_call_pa_delta, x_lo_call, x_hi_call,
                                                iters=max_iter)
    x_k_max = np.log(k_max)

    diff = delta - delta_max  # delta > 0 expected for calls
    no_sol_call = diff > tol_existence
    one_sol_call = np.abs(diff) <= tol_existence
    # two_sol_call = diff < -tol_existence

    k_call_left = _vectorized_bisect(_call_pa_delta, delta, x_lo_call, x_k_max, increasing=True, max_iter=max_iter)
    k_call_right = _vectorized_bisect(_call_pa_delta, delta, x_k_max, x_hi_call, increasing=False, max_iter=max_iter)

    if double_root_preference not in ("small", "large"):
        raise ValueError("double_root_preference must be 'small' or 'large'.")
    k_call_primary = k_call_left if double_root_preference == "small" else k_call_right
    k_call_secondary = k_call_right if double_root_preference == "small" else k_call_left
    k_call_primary = np.where(one_sol_call, k_max, k_call_primary)
    k_call_secondary = np.where(one_sol_call, np.nan, k_call_secondary)
    k_call_primary = np.where(no_sol_call, np.nan, k_call_primary)
    k_call_secondary = np.where(no_sol_call, np.nan, k_call_secondary)

    n_sol_call = np.where(no_sol_call, 0, np.where(one_sol_call, 1, 2))

    # Combine the three branches
    is_pa = prem_adjusted
    is_put = phi < 0
    is_call = ~is_put

    k = np.where(~is_pa, k_cf, np.where(is_put, k_put_pa, k_call_primary))
    k_alt = np.where(~is_pa, np.nan, np.where(is_put, np.nan, k_call_secondary))
    n_solutions = np.where(~is_pa, np.where(valid_cf, 1, 0), np.where(is_put, 1, n_sol_call)).astype(int)
    delta_max_abs = np.where(is_pa & is_call, delta_max, np.nan)

    # Final safety net: recompute delta at the chosen K and check it matches
    with np.errstate(invalid="ignore"):
        residual = bs_delta(f, k, sigma, t, phi, disc, prem_adjusted) - delta
    valid = np.where(~is_pa, valid_cf, True)
    valid = valid & (np.abs(residual) < tol_residual) & ~np.isnan(k)
    n_solutions = np.where(~valid & (n_solutions > 0), 0, n_solutions)
    k = np.where(valid, k, np.nan)

    return StrikeSolution(k=k, k_alt=k_alt, n_solutions=n_solutions, delta_max_abs=delta_max_abs, residual=residual,
                          valid=valid, used_spot_delta=use_spot_delta, prem_adjusted=prem_adjusted)


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("1) Round-trip on single quote, plain (non premium-adjusted)  delta put, EURUSD-like")
    print("=" * 70)
    delta = -0.25
    r_f, r_d, expiry = 0.02, 0.04, 0.5
    df_f, df_d = np.exp(-r_f * expiry), np.exp(-r_d * expiry)
    valdate = dt.datetime(2025, 12, 15)
    expiry = dt.datetime(2026, 12, 15)
    res = strike_from_delta(valdate, expiry, spot=1.10, df_f=df_f, df_d=df_d, sigma=0.09, delta=-delta,
                            option_type="P", prem_adjusted=False)
    print(res)
    # sanity check: recompute delta at that strike directly
    f = 1.10 * df_f / df_d
    d1, _ = _d1d2(np.array(f), res.k, np.array(0.09), np.array(0.5))
    print("check delta:", -norm.cdf(-d1) * np.exp(-0.02 * 0.5))

    print()
    print("=" * 70)
    print("2) Vectorized grid: multiple maturities (incl. one > 1Y) x multiple deltas,")
    print("   premium-adjusted, mixed put/call -- shows automatic spot/forward switch")
    print("=" * 70)
    # (3,1) -> broadcasts down columns
    maturities = np.array([dt.datetime(2026, 3, 15), dt.datetime(2026, 12, 15), dt.datetime(2027, 12, 15)])[:, None]
    # maturities = np.array([0.25, 1.0, 2.0])[:, None]
    deltas = np.array([-0.10, -0.25, 0.25, 0.10])[None, :] # (1,4) -> broadcasts across rows
    types = np.array([["P", "P", "C", "C"]] * 3)
    sigma_grid = np.array([[0.10, 0.095, 0.095, 0.105],
                           [0.11, 0.105, 0.105, 0.115],
                           [0.12, 0.115, 0.115, 0.125]])

    res_grid = strike_from_delta(valdate, maturities, spot=1.10, r_d=0.04, r_f=0.02, sigma=sigma_grid,
                                 delta=deltas, option_type=types, prem_adjusted=True)
    print("Strikes:\n", res_grid.k)
    print("Used spot-delta convention (True) vs forward-delta (False):\n",
          res_grid.used_spot_delta)
    print("n_solutions:\n", res_grid.n_solutions)
    print("valid:\n", res_grid.valid)

    print()
    print("=" * 70)
    print("3) Deliberately triggering NO-SOLUTION and DOUBLE-SOLUTION for a")
    print("   premium-adjusted call by sweeping the target delta past its max")
    print("=" * 70)
    sweep_deltas = np.linspace(0.01, 0.75, 15)
    res_sweep = strike_from_delta(valdate, expiry, spot=1.10, r_d=0.04, r_f=0.02, sigma=0.15, delta=sweep_deltas,
                                  option_type="C", prem_adjusted=True)

    for d, k, k_alt, n, dmax in zip(sweep_deltas, res_sweep.k, res_sweep.k_alt,
                                     res_sweep.n_solutions, res_sweep.delta_max_abs, strict=True):
        print(f"  target delta={d:5.3f} n_solutions={n} K={k:9.5f} K_alt={k_alt:9.5f} max achievable delta={dmax:.4f}")
