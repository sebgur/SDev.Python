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
from sdevpy.analytics import black
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta
from sdevpy.volatility.fx import fx_smilecalib


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
    spot: float = None
    df_f: float = None
    df_d: float = None
    prem_adjusted: bool = False

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
            solved = black.implied_vol(self.expiry, k_sub, flag, self.fwd, price)
            repriced = black.price(self.expiry, k_sub, flag, self.fwd, solved)
            vol[mask] = np.where(np.abs(repriced - price) < 1e-8, solved, np.nan)
        return vol

    def vol_at_delta(self, delta: float, is_call: bool, tol: float=1e-10, max_iter: int=100, **strike_kwargs) -> float:
        """ Vol at the given (unsigned) target delta -- the inverse of vol(strike).

            Since the strike for a given delta depends on the vol at that strike, and the vol depends on the smile
            evaluated at that (unknown) strike, there is no closed form. This is the classic FX smile-strike problem.
            We solve it here by fixed-point iteration: seed at atm_vol, find the strike for that vol via
            strike_from_delta, read the smile vol at that strike, repeat.

            Converges to machine precision in a handful of iterations for a smooth smile.

            Note: with extrapolation='flat' (the default), a delta whose strike falls outside [k_put, k_call] converges
            to the flat boundary vol, not a genuine extrapolated value. The smile has no information beyond its own
            quoted wings. """
        if self.spot is None or self.df_f is None or self.df_d is None:
            raise ValueError("spot/r_d/r_f not set on this smile -- build it via smile_from_quotes")

        strike_kwargs.setdefault('double_root_preference', 'large')
        signed_delta = delta if is_call else -delta
        sigma = self.atm_vol
        for _ in range(max_iter):
            sol = strike_from_delta(self.spot, self.df_f, self.df_d, self.expiry, sigma, signed_delta,
                                    'C' if is_call else 'P', prem_adjusted=self.prem_adjusted,
                                    **strike_kwargs)
            if not bool(np.asarray(sol.valid)):
                raise ValueError(f"No valid strike for delta={delta} at trial vol={sigma}")
            sigma_new = float(self.vol(float(sol.k)))
            if abs(sigma_new - sigma) < tol:
                return sigma_new
            sigma = sigma_new

        raise RuntimeError(f"vol_at_delta did not converge for delta={delta} after {max_iter} iterations")


def _smile_from_smile_butterfly(spot, df_f, df_d, expiry, atm_vol, rr, bf, delta, prem_adjusted, extrapolation,
                                **kwargs) -> VannaVolgaSmile:
    """ Build the smile treating `bf` as a smile (vol) butterfly """
    vol_call = atm_vol + bf + 0.5 * rr
    vol_put = atm_vol + bf - 0.5 * rr

    fwd = spot * df_f / df_d
    k_atm = atm_dns_strike(fwd, atm_vol, expiry, prem_adjusted)
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault('double_root_preference', 'large')

    sol_put = strike_from_delta(spot, df_f, df_d, expiry, vol_put, -delta, 'P', prem_adjusted=prem_adjusted, **kwargs)
    sol_call = strike_from_delta(spot, df_f, df_d, expiry, vol_call, delta, 'C', prem_adjusted=prem_adjusted,
                                 **call_kwargs)

    if not (np.all(sol_put.valid) and np.all(sol_call.valid)):
        raise ValueError(f"Could not solve pillar strikes for {delta}-delta quotes "
                         f"(put valid={sol_put.valid}, call valid={sol_call.valid})")

    return VannaVolgaSmile(fwd=float(fwd), expiry=float(expiry), k_put=float(sol_put.k),
                           k_atm=float(k_atm), k_call=float(sol_call.k), vol_put=float(vol_put),
                           atm_vol=float(atm_vol), vol_call=float(vol_call),
                           extrapolation=extrapolation, smile_butterfly=float(bf),
                           spot=float(spot), df_f=float(df_f), df_d=float(df_d),
                           prem_adjusted=bool(prem_adjusted))


def smile_from_quotes(spot: float, df_f: float, df_d: float, expiry: float, atm_vol: float, rr: float, bf: float,
                      delta: float=0.25, prem_adjusted: bool=False, extrapolation: str='flat',
                      market_strangle_quote: bool=False, **kwargs) -> VannaVolgaSmile:
    """ Vanna-Volga interpolation from quotes """
    # df_f, df_d = np.exp(-expiry * r_f), np.exp(-expiry * r_d)

    if market_strangle_quote:
        def build_smile(trial_bf):
            return _smile_from_smile_butterfly(spot, df_f, df_d, expiry, atm_vol, rr, trial_bf,
                                               delta, prem_adjusted, 'none', **kwargs)

        smile_bf = fx_smilecalib.calibrate_smile_strangle(spot, df_f, df_d, expiry, atm_vol, rr, bf, build_smile,
                                                          delta, prem_adjusted, **kwargs)
    else:
        smile_bf = bf

    smile = _smile_from_smile_butterfly(spot, df_f, df_d, expiry, atm_vol, rr, smile_bf, delta,
                                        prem_adjusted, extrapolation, **kwargs)

    if market_strangle_quote:
        smile.market_butterfly = float(bf)

    return smile


def wingvols_from_market_strangle_vv(spot: float, r_d: float, r_f: float, expiry: float, atm_vol: float,
                                     rr: float, ms: float, delta: float=0.25,
                                     prem_adjusted: bool=False, **kwargs) -> tuple:
    """ VV-specific: the only thing this adds over fx_smilecalib's generic version is the
        build_smile closure -- how to construct a VannaVolgaSmile from a candidate strangle. """
    df_f, df_d = np.exp(-expiry * r_f), np.exp(-expiry * r_d)

    def build_smile(trial_bf):
        return _smile_from_smile_butterfly(spot, df_f, df_d, expiry, atm_vol, rr, trial_bf, delta,
                                           prem_adjusted, 'none', **kwargs)

    return fx_smilecalib.wingvols_from_market_strangle(spot, df_f, df_d, expiry, atm_vol, rr, ms,
                                                       build_smile, delta, prem_adjusted, **kwargs)

if __name__ == "__main__":
    r_d, r_f = 0.04, 0.02
    expiry = 1.0
    df_f, df_d = np.exp(-expiry * r_f), np.exp(-expiry * r_d)
    s = smile_from_quotes(spot=1.10, df_f=df_f, df_d=df_d, expiry=expiry, atm_vol=0.10, rr=-0.01, bf=0.0025)
    print(f"Pillars: K={s.k_put:.4f}/{s.k_atm:.4f}/{s.k_call:.4f} vol={s.vol_put:.4f}/{s.atm_vol:.4f}/{s.vol_call:.4f}")
    for k in np.linspace(0.95, 1.35, 9):
        print(f"  K={k:.4f}  vv={float(s.vol(k)):.6f} 1st-order={float(s.vol(k, 'first_order')):.6f}")
