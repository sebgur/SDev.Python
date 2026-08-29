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
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from scipy.stats import norm

ArrayLike = Union[float, np.ndarray]

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _phi_from_option_type(option_type) -> np.ndarray:
    """Map 'C'/'P' (any case, numpy array of strings/objects) or +-1 to phi = +-1."""
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


def _d1d2(F: np.ndarray, K: np.ndarray, sigma: np.ndarray, T: np.ndarray):
    # K can legitimately hit the extreme edges of a search bracket (~exp(+-huge)),
    # which can underflow/overflow in float64; clip to keep log()/division finite.
    # This never affects the *reported* solution -- it only keeps the solver's
    # intermediate probing well-defined so we get a clean "no root here" signal
    # instead of a numpy warning.
    K = np.clip(K, 1e-300, 1e300)
    sqrtT = np.sqrt(T)
    vol_sqrtT = sigma * sqrtT
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / vol_sqrtT
        d2 = d1 - vol_sqrtT
    return d1, d2


def bs_delta(F, K, sigma, T, phi, disc, premium_adjusted) -> np.ndarray:
    """
    Vectorized Garman-Kohlhagen delta.

    `disc` is the multiplicative factor that distinguishes spot delta (disc = exp(-r_f*T))
    from forward delta (disc = 1). Passing the right `disc` array is how the
    spot/forward convention switch is implemented -- see `strike_from_delta`.
    """
    F, K, sigma, T, phi, disc = np.broadcast_arrays(
        *[_arr(a) for a in (F, K, sigma, T, phi, disc)]
    )
    premium_adjusted = np.broadcast_to(_arr(premium_adjusted).astype(bool), F.shape)
    d1, d2 = _d1d2(F, K, sigma, T)
    raw = phi * norm.cdf(phi * d1)
    pa = phi * (K / F) * norm.cdf(phi * d2)
    return np.where(premium_adjusted, pa, raw) * disc


# --------------------------------------------------------------------------- #
# Generic vectorized 1-D solvers (operate elementwise on whole arrays at once)
# --------------------------------------------------------------------------- #

def _vectorized_bisect(func, target, x_lo, x_hi, increasing: bool,
                        tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
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
    [exp(x_lo), exp(x_hi)], in log(K) space. Returns (K_argmax, func(K_argmax)).
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
    K_star = np.exp(x_star)
    return K_star, func(K_star)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class StrikeSolution:
    K: np.ndarray                 # recommended strike (NaN where no valid solution)
    K_alt: np.ndarray             # the *other* root when two exist, else NaN
    n_solutions: np.ndarray       # 0, 1, or 2 (int array)
    delta_max_abs: np.ndarray     # max achievable |premium-adjusted call delta|; NaN elsewhere
    residual: np.ndarray          # bs_delta(K) - target_delta, recomputed for the *chosen* K
    valid: np.ndarray             # bool: True iff `residual` is within tolerance (i.e. K is trustworthy)
    used_spot_delta: np.ndarray   # bool: True where the spot-delta convention was used (vs forward-delta)
    premium_adjusted: np.ndarray  # bool: echoed back, per element

    def __repr__(self):
        return (f"StrikeSolution(K={self.K}, n_solutions={self.n_solutions}, "
                f"valid={self.valid})")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def strike_from_delta(
    spot: ArrayLike,
    r_d: ArrayLike,
    r_f: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    delta: ArrayLike,
    option_type: ArrayLike,
    premium_adjusted: ArrayLike = False,
    spot_delta: Optional[ArrayLike] = None,
    spot_delta_cutoff: float = 1.0,
    double_root_preference: str = "small",
    bracket_width_sigma_mult: float = 15.0,
    bracket_width_floor: float = 8.0,
    tol_existence: float = 1e-9,
    tol_residual: float = 1e-6,
    max_iter: int = 100,
) -> StrikeSolution:
    """
    Invert Garman-Kohlhagen delta quotes into strikes. Fully vectorized: all
    array arguments are broadcast together (numpy broadcasting rules), so you can
    pass e.g. sigma/delta/option_type as a (n_maturities, n_deltas) grid and
    T/r_d/r_f as (n_maturities, 1), and get a (n_maturities, n_deltas) grid of
    strikes back in one call.

    Parameters
    ----------
    spot : spot FX rate S (domestic per foreign unit)
    r_d, r_f : domestic / foreign continuously-compounded risk-free rates
    T : time to expiry in years
    sigma : Black-Scholes volatility for that particular delta point (already
        the "smile" vol you'd plug into GK for that quote -- this function does
        the delta->strike leg only, not the smile interpolation itself)
    delta : the *signed* target delta (e.g. +0.25 for a 25-delta call, -0.25 for
        a 25-delta put), expressed in whichever convention (spot vs forward,
        premium-adjusted or not) is implied by the other flags below
    option_type : 'C'/'P' (any case) or +1/-1, broadcastable with the other inputs
    premium_adjusted : bool or bool array -- whether *each* quote uses
        premium-adjusted delta (currency-pair/market dependent; this function
        does not hardcode a currency-pair table since conventions can change --
        pass it explicitly per quote or as a single bool for all quotes)
    spot_delta : optional bool/bool-array override. If None (default), the
        convention is chosen automatically per element: spot delta for
        T <= spot_delta_cutoff, forward delta for T > spot_delta_cutoff. This
        is the standard market rule (spot delta becomes a poor hedge-ratio
        proxy for long-dated options once forward points dominate).
    spot_delta_cutoff : maturity (in years) at which the convention switches
        (1.0 = 1Y, the market standard; override if needed for a specific pair)
    double_root_preference : 'small' (default, market convention) or 'large' --
        which of the two roots to report as `K` when a premium-adjusted call
        has two solutions. Both roots are always available via `K_alt`.
    bracket_width_sigma_mult, bracket_width_floor : control how wide (in units
        of log-strike) the numerical search brackets are. The defaults are
        generous (many sigma*sqrt(T) wide) and should not need changing.
    tol_existence : tolerance (in delta units) used to classify
        "no solution" vs "exactly one" vs "two solutions" for premium-adjusted
        calls, relative to the achievable maximum delta.
    tol_residual : after solving, the delta implied by the returned strike is
        recomputed and compared to the target; if the discrepancy exceeds this,
        the element is marked invalid regardless of how it was classified.
    max_iter : iterations for the bisection / ternary-search solvers.

    Returns
    -------
    StrikeSolution
    """
    # ---- broadcast everything ---------------------------------------------------
    S, r_d, r_f, T, sigma, delta = np.broadcast_arrays(
        *[_arr(a) for a in (spot, r_d, r_f, T, sigma, delta)]
    )
    phi = np.broadcast_to(_phi_from_option_type(option_type), S.shape).astype(float)
    premium_adjusted = np.broadcast_to(_arr(premium_adjusted).astype(bool), S.shape)

    if np.any(sigma <= 0) or np.any(T <= 0):
        raise ValueError("sigma and T must be strictly positive everywhere.")

    F = S * np.exp((r_d - r_f) * T)
    df_f = np.exp(-r_f * T)

    if spot_delta is None:
        use_spot_delta = T <= spot_delta_cutoff
    else:
        use_spot_delta = np.broadcast_to(_arr(spot_delta).astype(bool), S.shape)
    disc = np.where(use_spot_delta, df_f, 1.0)

    shape = S.shape
    sqrtT = np.sqrt(T)
    B = bracket_width_sigma_mult * sigma * sqrtT + bracket_width_floor  # log-K half-width

    # ================================================================
    # Branch 1: plain (non premium-adjusted) delta -> closed form
    # ================================================================
    #   delta = phi * disc * N(phi*d1)   =>   d1 = phi * N^{-1}(phi*delta/disc)
    x_cf = phi * delta / disc
    with np.errstate(invalid="ignore"):
        d1_cf = phi * norm.ppf(x_cf)  # NaN automatically outside (0,1) -- that's correct: no solution
    K_cf = F * np.exp(-d1_cf * sigma * sqrtT + 0.5 * sigma ** 2 * T)
    valid_cf = (x_cf > 0) & (x_cf < 1)

    # ================================================================
    # Branch 2: premium-adjusted PUT delta -> monotonic, unique root
    # ================================================================
    def _put_pa_delta(K):
        _, d2 = _d1d2(F, K, sigma, T)
        return -disc * (K / F) * norm.cdf(-d2)

    x_lo_put = np.log(F) - B
    x_hi_put = np.log(F) + B
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

    K_put_pa = _vectorized_bisect(_put_pa_delta, delta, x_lo_put, x_hi_put,
                                   increasing=False, max_iter=max_iter)

    # ================================================================
    # Branch 3: premium-adjusted CALL delta -> unimodal (hump), 0/1/2 roots
    # ================================================================
    def _call_pa_delta(K):
        _, d2 = _d1d2(F, K, sigma, T)
        return disc * (K / F) * norm.cdf(d2)

    x_lo_call = np.log(F) - B
    x_hi_call = np.log(F) + B
    K_max, delta_max = _vectorized_ternary_max(_call_pa_delta, x_lo_call, x_hi_call,
                                                iters=max_iter)
    x_Kmax = np.log(K_max)

    diff = delta - delta_max  # delta > 0 expected for calls
    no_sol_call = diff > tol_existence
    one_sol_call = np.abs(diff) <= tol_existence
    two_sol_call = diff < -tol_existence

    K_call_left = _vectorized_bisect(_call_pa_delta, delta, x_lo_call, x_Kmax,
                                      increasing=True, max_iter=max_iter)
    K_call_right = _vectorized_bisect(_call_pa_delta, delta, x_Kmax, x_hi_call,
                                       increasing=False, max_iter=max_iter)

    if double_root_preference not in ("small", "large"):
        raise ValueError("double_root_preference must be 'small' or 'large'.")
    K_call_primary = K_call_left if double_root_preference == "small" else K_call_right
    K_call_secondary = K_call_right if double_root_preference == "small" else K_call_left
    K_call_primary = np.where(one_sol_call, K_max, K_call_primary)
    K_call_secondary = np.where(one_sol_call, np.nan, K_call_secondary)
    K_call_primary = np.where(no_sol_call, np.nan, K_call_primary)
    K_call_secondary = np.where(no_sol_call, np.nan, K_call_secondary)

    n_sol_call = np.where(no_sol_call, 0, np.where(one_sol_call, 1, 2))

    # ================================================================
    # Combine the three branches
    # ================================================================
    is_pa = premium_adjusted
    is_put = phi < 0
    is_call = ~is_put

    K = np.where(~is_pa, K_cf,
                 np.where(is_put, K_put_pa, K_call_primary))
    K_alt = np.where(~is_pa, np.nan,
                      np.where(is_put, np.nan, K_call_secondary))
    n_solutions = np.where(~is_pa, np.where(valid_cf, 1, 0),
                            np.where(is_put, 1, n_sol_call)).astype(int)
    delta_max_abs = np.where(is_pa & is_call, delta_max, np.nan)

    # ---- final safety net: recompute delta at the chosen K and check it matches ----
    with np.errstate(invalid="ignore"):
        residual = bs_delta(F, K, sigma, T, phi, disc, premium_adjusted) - delta
    valid = np.where(~is_pa, valid_cf, True)
    valid = valid & (np.abs(residual) < tol_residual) & ~np.isnan(K)
    n_solutions = np.where(~valid & (n_solutions > 0), 0, n_solutions)
    K = np.where(valid, K, np.nan)

    return StrikeSolution(
        K=K,
        K_alt=K_alt,
        n_solutions=n_solutions,
        delta_max_abs=delta_max_abs,
        residual=residual,
        valid=valid,
        used_spot_delta=use_spot_delta,
        premium_adjusted=premium_adjusted,
    )


# --------------------------------------------------------------------------- #
# Demo / self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    print("=" * 70)
    print("1) Single quote, plain (non premium-adjusted) 25-delta put, EURUSD-like")
    print("=" * 70)
    res = strike_from_delta(
        spot=1.10, r_d=0.04, r_f=0.02, T=0.5, sigma=0.09,
        delta=-0.25, option_type="P", premium_adjusted=False,
    )
    print(res)
    # sanity check: recompute delta at that strike directly
    F = 1.10 * np.exp((0.04 - 0.02) * 0.5)
    d1, _ = _d1d2(np.array(F), res.K, np.array(0.09), np.array(0.5))
    print("check delta:", -norm.cdf(-d1) * np.exp(-0.02 * 0.5))

    print()
    print("=" * 70)
    print("2) Vectorized grid: multiple maturities (incl. one > 1Y) x multiple deltas,")
    print("   premium-adjusted, mixed put/call -- shows automatic spot/forward switch")
    print("=" * 70)
    maturities = np.array([0.25, 1.0, 2.0])[:, None]          # (3,1) -> broadcasts down columns
    deltas = np.array([-0.10, -0.25, 0.25, 0.10])[None, :]     # (1,4) -> broadcasts across rows
    types = np.array([["P", "P", "C", "C"]] * 3)
    sigma_grid = np.array([[0.10, 0.095, 0.095, 0.105],
                            [0.11, 0.105, 0.105, 0.115],
                            [0.12, 0.115, 0.115, 0.125]])

    res_grid = strike_from_delta(
        spot=1.10, r_d=0.04, r_f=0.02, T=maturities, sigma=sigma_grid,
        delta=deltas, option_type=types, premium_adjusted=True,
    )
    print("Strikes:\n", res_grid.K)
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
    res_sweep = strike_from_delta(
        spot=1.10, r_d=0.04, r_f=0.02, T=1.5, sigma=0.15,
        delta=sweep_deltas, option_type="C", premium_adjusted=True,
    )
    for d, k, k_alt, n, dmax in zip(sweep_deltas, res_sweep.K, res_sweep.K_alt,
                                     res_sweep.n_solutions, res_sweep.delta_max_abs):
        print(f"  target delta={d:5.3f}  n_solutions={n}  K={k:9.5f}  "
              f"K_alt={k_alt:9.5f}  max achievable delta={dmax:.4f}")