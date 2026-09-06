""" Smile-strangle calibration: converting a broker's quoted/market strangle into the
    smile-consistent correction that reprices it, independent of which smile-interpolation
    model is used.

Terminology (Reiswich & Wystup, "FX Volatility Smile Construction"):
    quoted strangle  (input `ms`)   -- the raw number printed on the broker's screen
    market strangle  (`atm_vol+ms`) -- the flat vol used to find both strikes and price the
                                       package; this is the "market strangle" proper
    smile strangle   (calibrated)   -- the corrected number such that a real smile function,
                                       evaluated at its own vols at those same strikes,
                                       reprices the same market-strangle premium
"""
import numpy as np
from scipy.optimize import brentq
from sdevpy.analytics import black
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta
from sdevpy.market.fxvolsurface import wingvols_from_butterfly


def market_strangle(spot: float, df_f: float, df_d: float, expiry: float, atm_vol: float, ms: float,
                    delta: float=0.25, prem_adjusted: bool=False, **kwargs) -> tuple:
    """ Resolve the broker's quoted strangle into the market-strangle price. Both wing strikes
        are struck off the single flat vol atm_vol+ms; the quote is the sum of the two option
        premia at that vol. Returns (k_put, k_call, fwd_price, vol_ms). """
    vol_ms = atm_vol + ms
    fwd = spot * df_f / df_d
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault('double_root_preference', 'large')

    sol_put = strike_from_delta(spot, df_f, df_d, expiry, vol_ms, -delta, 'P', prem_adjusted=prem_adjusted, **kwargs)
    sol_call = strike_from_delta(spot, df_f, df_d, expiry, vol_ms, delta, 'C', prem_adjusted=prem_adjusted, **call_kwargs)
    if not (np.all(sol_put.valid) and np.all(sol_call.valid)):
        raise ValueError(f"Could not solve market-strangle strikes at {delta}-delta "
                         f"(put valid={sol_put.valid}, call valid={sol_call.valid})")

    k_put, k_call = float(sol_put.k), float(sol_call.k)
    price = float(black.price(expiry, k_call, True, fwd, vol_ms) + black.price(expiry, k_put, False, fwd, vol_ms))
    return k_put, k_call, price, vol_ms


def calibrate_smile_strangle(spot: float, df_f: float, df_d: float, expiry: float, atm_vol: float, rr: float, ms: float,
                             build_smile, delta: float=0.25, prem_adjusted: bool=False, tol: float=1e-12,
                             max_expand: int=60, **kwargs) -> float:
    """ Find the strangle such that build_smile(strangle) reprices the market strangle at its own two strikes.

        build_smile: callable(bf) -> object with a .vol(strike) method (e.g. a VannaVolgaSmile, or any other
                     interpolation model built from (atm_vol, rr, bf)). This is the only point of contact with
                     any specific smile model. The calibration logic itself has none.

        Property: when rr == 0 the pillar strikes coincide with the market-strangle strikes, so this returns ms. """
    ####
    # df_f, df_d = np.exp(-expiry * r_f), np.exp(-expiry * r_d)
    ####

    k_put_ms, k_call_ms, target, _ = market_strangle(spot, df_f, df_d, expiry, atm_vol, ms, delta, prem_adjusted,
                                                     **kwargs)
    fwd = spot * df_f / df_d

    def objective(bf):
        smile = build_smile(bf)
        vol_put, vol_call = float(smile.vol(k_put_ms)), float(smile.vol(k_call_ms))
        if not (np.isfinite(vol_put) and np.isfinite(vol_call)):
            raise ValueError(f"Smile is not arbitrage-free at the market-strangle strikes for smile strangle {bf}")

        price = float(black.price(expiry, k_call_ms, True, fwd, vol_call) +
                      black.price(expiry, k_put_ms, False, fwd, vol_put))
        return price - target

    f_ms = objective(ms)
    if abs(f_ms) < 1e-16:
        return float(ms)

    bf_floor = 0.5 * abs(rr) - atm_vol + 1e-6
    lo = hi = float(ms)
    f_lo = f_hi = f_ms
    step, bracketed = 1e-3, False
    for _ in range(max_expand):
        trial = hi + step if f_ms < 0.0 else max(lo - step, bf_floor)
        if trial in (lo, hi):
            break
        try:
            f_trial = objective(trial)
        except ValueError:
            step *= 0.5
            if step < 1e-10:
                break
            continue
        if f_ms < 0.0:
            hi, f_hi = trial, f_trial
            bracketed = f_hi > 0.0
        else:
            lo, f_lo = trial, f_trial
            bracketed = f_lo < 0.0
        if bracketed:
            break
        step *= 2.0

    if not bracketed:
        raise ValueError(f"Could not bracket the smile strangle for atm={atm_vol}, rr={rr}, ms={ms}")

    return float(brentq(objective, lo, hi, xtol=tol))


def wingvols_from_market_strangle(spot: float, r_d: float, r_f: float, expiry: float, atm_vol: float,
                                  rr: float, ms: float, build_smile, delta: float = 0.25,
                                  prem_adjusted: bool = False, **kwargs) -> tuple:
    """ Call/put vols at the given delta, given atm_vol/rr/ms where ms is the broker's raw
        quoted/market strangle, not yet a smile strangle. Model-agnostic: build_smile is the
        only model-specific input -- see calibrate_smile_strangle's own contract. """
    ####
    df_f, df_d = np.exp(-expiry * r_f), np.exp(-expiry * r_d)
    ####
    smile_strangle = calibrate_smile_strangle(spot, df_f, df_d, expiry, atm_vol, rr, ms, build_smile,
                                              delta, prem_adjusted, **kwargs)
    return wingvols_from_butterfly(atm_vol, rr, smile_strangle)
