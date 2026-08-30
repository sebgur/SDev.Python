""" Vanna-Volga (VV) smile construction for FX options.

The market quotes three instruments per expiry: ATM, 25-delta RR and 25-delta BF, fixing the vol at three strikes.
VV builds the vol at any other strike from the cost of hedging the target's vega, vanna and volga with those three
traded instruments.

No calibration/optimizer: the hedge weights are closed form (Castagna & Mercurio, Risk 2007): quadratic Lagrange
coefficients in log-strike scaled by vega ratios. At K = K_i they collapse to the Kronecker delta, so the construction
reprices the three quotes exactly (verified to 1e-16).

Methods:
  'exact': build the VV price, invert numerically to a vol (the definition).
  'first_order': leading term only: quadratic Lagrange interpolation of the three vols in log-strike. No root search,
                 agrees with 'exact' to ~1e-5 inside the quoted range, degrades in the wings.

Extrapolation: beyond the outer pillars the formula is unconstrained, so extrapolation='flat' (default) holds vol at
the pillar level outside [k_put, k_call]. 'none' runs the raw formula.

Premium-adjusted note: the PA call delta is not monotonic in K, so a 25-delta PA call has two strikes. Only the OTM
(larger) root is a meaningful smile pillar, the smaller root is deep ITM (e.g. 0.286 against a 1.122 forward) and
produces non-monotonic pillars. This module therefore requests double_root_preference='large' by default and passes
it explicitly to override.
"""
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.optimize import brentq
from sdevpy.analytics import black
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta


def _arr(x) -> npt.NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def bs_vega(fwd: npt.ArrayLike, strike: npt.ArrayLike, expiry: npt.ArrayLike,
            vol: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """ Undiscounted (forward) Black vega. Identical for calls and puts. """
    fwd, strike, expiry, vol = (_arr(a) for a in (fwd, strike, expiry, vol))
    sqrt_t = np.sqrt(expiry)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(fwd / strike) + 0.5 * vol ** 2 * expiry) / (vol * sqrt_t)
    return fwd * norm.pdf(d1) * sqrt_t


def lagrange_weights(strike: npt.ArrayLike, k_put: float, k_atm: float, k_call: float) -> tuple:
    """ Quadratic Lagrange coefficients in log-strike. Sum to 1; Kronecker delta at the nodes. """
    k = _arr(strike)
    l12, l13, l23 = np.log(k_atm / k_put), np.log(k_call / k_put), np.log(k_call / k_atm)
    w1 = (np.log(k_atm / k) * np.log(k_call / k)) / (l12 * l13)
    w2 = (np.log(k / k_put) * np.log(k_call / k)) / (l12 * l23)
    w3 = (np.log(k / k_put) * np.log(k / k_atm)) / (l13 * l23)
    return w1, w2, w3


def vv_weights(strike: npt.ArrayLike, k_put: float, k_atm: float, k_call: float,
               fwd: float, expiry: float, atm_vol: float) -> tuple:
    """ Castagna-Mercurio vega/vanna/volga hedge weights """
    k = _arr(strike)
    w1, w2, w3 = lagrange_weights(k, k_put, k_atm, k_call)
    vega = bs_vega(fwd, k, expiry, atm_vol)
    v1 = bs_vega(fwd, k_put, expiry, atm_vol)
    v2 = bs_vega(fwd, k_atm, expiry, atm_vol)
    v3 = bs_vega(fwd, k_call, expiry, atm_vol)
    return w1 * vega / v1, w2 * vega / v2, w3 * vega / v3


def _implied_vol_bisect(fwd_price: npt.ArrayLike, fwd: float, strike: npt.ArrayLike, expiry: float, is_call: bool,
                        vol_lo: float = 1e-8, vol_hi: float = 5.0, max_iter: int = 200,
                        tol: float = 1e-14) -> npt.NDArray[np.float64]:
    """ Invert an undiscounted forward price to a Black vol by bisection.
        Deliberately not black.implied_vol_newton: Newton divides by vega, which collapses in
        the deep wings where a VV smile is most often queried, and returns NaN silently. Price
        is strictly increasing in vol, so bisection cannot diverge. """
    target = _arr(fwd_price)
    k = np.broadcast_to(_arr(strike), target.shape)
    lo = np.full(target.shape, vol_lo)
    hi = np.full(target.shape, vol_hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        go_up = (black.price(expiry, k, is_call, fwd, mid) - target) < 0.0
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
        if np.nanmax(hi - lo) < tol:
            break
    return 0.5 * (lo + hi)


def atm_dns_strike(fwd: float, atm_vol: float, expiry: float, prem_adjusted: bool = False) -> float:
    """ Delta-neutral-straddle ATM strike (the FX convention, not ATM-forward).
        Non premium-adjusted: F exp(+0.5 sigma^2 T); premium-adjusted: F exp(-0.5 sigma^2 T). """
    sign = -1.0 if prem_adjusted else 1.0
    return fwd * np.exp(sign * 0.5 * atm_vol ** 2 * expiry)


@dataclass
class VannaVolgaSmile:
    """ Single-expiry FX smile pinned to three (strike, vol) pillars. Returns NaN where the VV
        price falls outside the no-arbitrage range, rather than a bisection boundary. """
    fwd: float
    expiry: float
    k_put: float
    k_atm: float
    k_call: float
    vol_put: float
    atm_vol: float
    vol_call: float
    extrapolation: str = 'flat'
    smile_butterfly: float = None
    market_butterfly: float = None

    def __post_init__(self):
        if self.extrapolation not in ('flat', 'none'):
            raise ValueError(f"extrapolation must be 'flat' or 'none', got: {self.extrapolation}")
        if not self.k_put < self.k_atm < self.k_call:
            raise ValueError(f"Pillar strikes must be increasing, got: "
                             f"{self.k_put}, {self.k_atm}, {self.k_call}")

    def price(self, strike: npt.ArrayLike, is_call: bool=True) -> npt.NDArray[np.float64]:
        """ VV-adjusted undiscounted forward price (multiply by exp(-r_d T) to settle) """
        k = _arr(strike)
        x1, x2, x3 = vv_weights(k, self.k_put, self.k_atm, self.k_call,
                                self.fwd, self.expiry, self.atm_vol)
        base = black.price(self.expiry, k, is_call, self.fwd, self.atm_vol)

        def _corr(k_i, vol_i):
            return (black.price(self.expiry, k_i, is_call, self.fwd, vol_i)
                    - black.price(self.expiry, k_i, is_call, self.fwd, self.atm_vol))

        # The K_atm term is identically zero (vol == atm_vol); kept for symmetry.
        return (base + x1 * _corr(self.k_put, self.vol_put)
                + x2 * _corr(self.k_atm, self.atm_vol)
                + x3 * _corr(self.k_call, self.vol_call))

    def vol(self, strike: npt.ArrayLike, method: str='exact') -> npt.NDArray[np.float64]:
        """ Smile vol at the given strike(s) """
        k = _arr(strike)
        if method == 'first_order':
            result = self.first_order_vol(k)
        elif method == 'exact':
            result = self._exact_vol(k)
        else:
            raise ValueError(f"method must be 'exact' or 'first_order', got: {method}")

        if self.extrapolation == 'flat':
            result = np.where(k <= self.k_put, self.vol_put, result)
            result = np.where(k >= self.k_call, self.vol_call, result)
        return result

    def first_order_vol(self, strike: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """ Leading-order VV: quadratic Lagrange interpolation of the three vols in log-strike """
        w1, w2, w3 = lagrange_weights(strike, self.k_put, self.k_atm, self.k_call)
        return w1 * self.vol_put + w2 * self.atm_vol + w3 * self.vol_call

    def _exact_vol(self, k: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Calls and puts imply the same vol (the VV correction is identical by put-call parity),
        # so invert whichever is OTM -- better conditioned.
        is_call = np.asarray(k >= self.fwd)
        vol = np.empty(np.broadcast(k, is_call).shape, dtype=float)
        for flag in (True, False):
            mask = (is_call == flag)
            if not np.any(mask):
                continue
            k_sub = np.broadcast_to(k, vol.shape)[mask]
            price = self.price(k_sub, is_call=flag)
            if flag:
                intrinsic = np.maximum(self.fwd - k_sub, 0.0)
                upper = np.full_like(k_sub, self.fwd)
            else:
                intrinsic = np.maximum(k_sub - self.fwd, 0.0)
                upper = k_sub
            solved = _implied_vol_bisect(price, self.fwd, k_sub, self.expiry, flag)
            vol[mask] = np.where((price > intrinsic) & (price < upper), solved, np.nan)
        return vol


def market_strangle(spot: float, r_d: float, r_f: float, expiry: float, atm_vol: float, ms: float,
                    delta: float = 0.25, prem_adjusted: bool=False, **kwargs) -> tuple:
    """ Resolve the broker's market strangle: a *price*, not a vol. Both wing strikes are struck
        off the single volatility atm_vol + ms, and the quote is the sum of the two premia at
        that vol. Returns (k_put, k_call, fwd_price, vol_ms); the strikes stay fixed during
        the smile-butterfly calibration. """
    vol_ms = atm_vol + ms
    fwd = spot * np.exp((r_d - r_f) * expiry)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault('double_root_preference', 'large')

    sol_put = strike_from_delta(spot, r_d, r_f, expiry, vol_ms, -delta, 'P',
                                prem_adjusted=prem_adjusted, **kwargs)
    sol_call = strike_from_delta(spot, r_d, r_f, expiry, vol_ms, delta, 'C',
                                 prem_adjusted=prem_adjusted, **call_kwargs)
    if not (np.all(sol_put.valid) and np.all(sol_call.valid)):
        raise ValueError(f"Could not solve market-strangle strikes at {delta}-delta "
                         f"(put valid={sol_put.valid}, call valid={sol_call.valid})")

    k_put, k_call = float(sol_put.k), float(sol_call.k)
    price = float(black.price(expiry, k_call, True, fwd, vol_ms)
                  + black.price(expiry, k_put, False, fwd, vol_ms))
    return k_put, k_call, price, vol_ms


def _smile_from_smile_butterfly(spot, r_d, r_f, expiry, atm_vol, rr, bf, delta, prem_adjusted, extrapolation,
                                **kwargs) -> VannaVolgaSmile:
    """ Build the smile treating `bf` as a smile (vol) butterfly """
    vol_call = atm_vol + bf + 0.5 * rr
    vol_put = atm_vol + bf - 0.5 * rr

    fwd = spot * np.exp((r_d - r_f) * expiry)
    k_atm = atm_dns_strike(fwd, atm_vol, expiry, prem_adjusted)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault('double_root_preference', 'large')

    sol_put = strike_from_delta(spot, r_d, r_f, expiry, vol_put, -delta, 'P',
                                prem_adjusted=prem_adjusted, **kwargs)
    sol_call = strike_from_delta(spot, r_d, r_f, expiry, vol_call, delta, 'C',
                                 prem_adjusted=prem_adjusted, **call_kwargs)
    if not (np.all(sol_put.valid) and np.all(sol_call.valid)):
        raise ValueError(f"Could not solve pillar strikes for {delta}-delta quotes "
                         f"(put valid={sol_put.valid}, call valid={sol_call.valid})")

    return VannaVolgaSmile(fwd=float(fwd), expiry=float(expiry), k_put=float(sol_put.k),
                           k_atm=float(k_atm), k_call=float(sol_call.k), vol_put=float(vol_put),
                           atm_vol=float(atm_vol), vol_call=float(vol_call),
                           extrapolation=extrapolation, smile_butterfly=float(bf))


def calibrate_smile_butterfly(spot: float, r_d: float, r_f: float, expiry: float, atm_vol: float, rr: float,
                              ms: float, delta: float = 0.25, prem_adjusted: bool = False, tol: float = 1e-12,
                              max_expand: int = 60, **kwargs) -> float:
    """ Convert a broker market-strangle quote into the smile butterfly that reproduces it.

        Solves for sigma_bf such that the VV smile built from (atm_vol, rr, sigma_bf) reprices
        the market strangle at its own two fixed strikes. The objective is monotonically
        increasing in sigma_bf, so the bracket-and-solve cannot land on a spurious root.

        Verified property: when rr == 0 the pillar strikes coincide with the market-strangle
        strikes, so this returns ms unchanged (to 1e-15). """
    k_put_ms, k_call_ms, target, _ = market_strangle(spot, r_d, r_f, expiry, atm_vol, ms,
                                                     delta, prem_adjusted, **kwargs)
    fwd = spot * np.exp((r_d - r_f) * expiry)

    def objective(bf):
        # extrapolation='none': the market-strangle strikes can sit marginally outside the
        # pillar strikes, and flat extrapolation there would stall the solve.
        smile = _smile_from_smile_butterfly(spot, r_d, r_f, expiry, atm_vol, rr, bf, delta,
                                            prem_adjusted, 'none', **kwargs)
        vol_put, vol_call = float(smile.vol(k_put_ms)), float(smile.vol(k_call_ms))
        if not (np.isfinite(vol_put) and np.isfinite(vol_call)):
            raise ValueError(f"VV smile is not arbitrage-free at the market-strangle strikes "
                             f"for butterfly {bf}")
        price = float(black.price(expiry, k_call_ms, True, fwd, vol_call)
                      + black.price(expiry, k_put_ms, False, fwd, vol_put))
        return price - target

    # Anchor at bf = ms: always well-posed (it is the broker's quote, and the exact answer when
    # rr == 0), with the root within a few 1e-3. Walk outwards in the direction the residual
    # points, halving the step if a probe lands where the smile is not arbitrage-free.
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
        raise ValueError(f"Could not bracket the smile butterfly for atm={atm_vol}, rr={rr}, ms={ms}")

    return float(brentq(objective, lo, hi, xtol=tol))


def smile_from_quotes(spot: float, r_d: float, r_f: float, expiry: float, atm_vol: float, rr: float, bf: float,
                      delta: float=0.25, prem_adjusted: bool=False, extrapolation: str='flat',
                      market_strangle_quote: bool=False, **kwargs) -> VannaVolgaSmile:
    """ Build a VV smile from the standard FX quote triple.

        rr = vol_call - vol_put (negative means a put skew, typical for EURUSD)
        delta: the delta the RR/BF are quoted at (0.25 or 0.10)

        market_strangle_quote:
            False (default): `bf` is a *smile* butterfly, used directly.
            True: `bf` is the broker's *market* strangle; the smile butterfly is solved for so the smile reprices that
                  strangle. Costs a root-find per smile. The correction is ~0.3bp for rr=-0.5%, ~1.5bp for rr=-1%,
                  ~28bp for rr=-4%, so it matters most for wide smiles and 10-delta quotes.
    """
    smile_bf = (calibrate_smile_butterfly(spot, r_d, r_f, expiry, atm_vol, rr, bf, delta, prem_adjusted, **kwargs)
                if market_strangle_quote else bf)

    smile = _smile_from_smile_butterfly(spot, r_d, r_f, expiry, atm_vol, rr, smile_bf, delta,
                                        prem_adjusted, extrapolation, **kwargs)
    if market_strangle_quote:
        smile.market_butterfly = float(bf)
    return smile


if __name__ == "__main__":
    s = smile_from_quotes(spot=1.10, r_d=0.04, r_f=0.02, expiry=1.0, atm_vol=0.10, rr=-0.01, bf=0.0025)
    print(f"Pillars: K={s.k_put:.4f}/{s.k_atm:.4f}/{s.k_call:.4f} vol={s.vol_put:.4f}/{s.atm_vol:.4f}/{s.vol_call:.4f}")
    for k in np.linspace(0.95, 1.35, 9):
        print(f"  K={k:.4f}  vv={float(s.vol(k)):.6f} 1st-order={float(s.vol(k, 'first_order')):.6f}")
