""" Explicit Black-Scholes implied volatility via the inverse-Gaussian quantile, following
    W. Schadner, "An Explicit Solution to Black-Scholes Implied Volatility" (arXiv:2604.24480, 2026).

    The only non-elementary operation is the inverse-Gaussian (Wald) quantile F_IG^{-1}(q; mu, lambda).
    Design choices for speed AND accuracy:
    * The quantile is computed self-contained: a monotone analytic bracket on the closed-form CDF,
      then a safeguarded Halley iteration (bisection fallback) to machine precision. This is ~3-4x
      faster than seeding from scipy.stats.invgauss.ppf, which is itself a Boost root finder and the
      actual bottleneck (pass use_scipy_seed=True to compare).
    * The CDF is evaluated stably in the deep tail, where mu = 2/|k| -> 0 makes the exp(2*lambda/mu)
      term overflow: that term is formed in log space via scipy.special.log_ndtr. scipy supplies the
      Gaussian special functions (ndtr, log_ndtr, ndtri) -- the right tool for those.
    * Active-set compression retires converged points each iteration, so a few slow (near-boundary)
      points do not hold up the whole vectorised batch.

    Accuracy/conditioning: recovery is machine-precision when the out-of-the-money normalized price
    exceeds ~1e-6. Inverting a deep in-the-money price is ill-conditioned (vol information lives in
    a tiny time-value lost to rounding). Prices at or beyond the no-arbitrage boundary return NaN.
    For best accuracy, invert the OTM wing (implied_vol_call for K>=F, implied_vol_put for K<=F).

    ndtr, log_ndtr and ndtri are used for extreme performance saving. They save by avoiding overheads
    for argument validation, error checking, etc. Recommended here as speed is the target.
"""
from __future__ import annotations
import warnings
import numpy as np
import numpy.typing as npt
from scipy.stats import invgauss
from scipy.special import ndtr, log_ndtr, ndtri


__all__ = ['implied_vol_schadner']


_ATM_TOL = 1e-12 # |k| below this is treated as at-the-forward


# Inverse-Gaussian (Wald) primitives, mean parameter `mu`, shape `lam`.
# Stable in the tails: the exp(2*lam/mu) * Phi(b) term is formed in log space.
def ig_cdf(x, mu, lam=1.0):
    """CDF of IG(mean=mu, shape=lam) evaluated at x (paper Eq. 4 complement)."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sl = np.sqrt(lam)
    a = -np.sqrt(lam * x) / mu + sl / np.sqrt(x)
    b = -np.sqrt(lam * x) / mu - sl / np.sqrt(x)
    surv = ndtr(a) - np.exp(2.0 * lam / mu + log_ndtr(b))
    return 1.0 - surv


def ig_logpdf(x, mu, lam=1.0):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return 0.5 * np.log(lam / (2.0 * np.pi)) - 1.5 * np.log(x) \
        - lam * (x - mu) ** 2 / (2.0 * mu ** 2 * x)


def ig_pdf(x, mu, lam=1.0):
    return np.exp(ig_logpdf(x, mu, lam))


def _ig_logpdf_deriv_factor(x, mu, lam=1.0):
    """d/dx log f(x)  =  -3/(2x) - lam/(2 mu^2) + lam/(2 x^2)."""
    return -1.5 / x - lam / (2.0 * mu ** 2) + lam / (2.0 * x ** 2)


def _scipy_seed(q, mu, lam):
    """ Fast seed from scipy's Boost quantile; validated, warnings suppressed.
        Boost can fail to converge in the extreme tail, so the result is only
        used where it is finite and positive -- elsewhere we fall back to the mean.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            s = invgauss.ppf(q, mu=mu / lam, scale=1.0) * lam
        except Exception:
            s = np.full_like(np.asarray(q, dtype=float), np.nan)
    s = np.asarray(s, dtype=float)
    return np.where(np.isfinite(s) & (s > 0.0), s, np.asarray(mu, dtype=float))


def ig_ppf(q, mu, lam=1.0, maxiter=60, tol=1e-15, use_scipy_seed=False):
    """ Inverse-Gaussian quantile F_IG^{-1}(q; mu, lam), vectorised.
        Self-contained and robust: a monotone analytic bracket is established, then a safeguarded Halley iteration
        (with bisection fallback) drives every point to machine precision. The default analytic seed is
        ~3-4x faster than seeding from scipy's Boost quantile (which is itself a root finder); set
        ``use_scipy_seed=True`` to compare. scipy is still used for the underlying Gaussian special functions,
        which is the right tool for those.
    """
    q, mu = np.broadcast_arrays(np.asarray(q, float), np.asarray(mu, float))
    shape = q.shape
    q = q.ravel().copy()
    mu = mu.ravel().copy()

    out = np.empty_like(q)
    # degenerate probabilities -> degenerate quantiles
    lo_deg = q <= 0.0
    hi_deg = q >= 1.0
    out[lo_deg] = 0.0
    out[hi_deg] = np.inf
    work = ~(lo_deg | hi_deg)
    if not np.any(work):
        return out.reshape(shape)

    qq, mm = q[work], mu[work]

    seed = _scipy_seed(qq, mm, lam) if use_scipy_seed else mm.copy()
    seed = np.where(seed > 0, seed, mm)

    # --- monotone bracket: cdf(lo) <= q <= cdf(hi); expand only stragglers ---
    hi = np.maximum(seed, mm)
    act = ig_cdf(hi, mm, lam) < qq
    for _ in range(80):
        if not act.any():
            break
        hi[act] *= 2.0
        act[act] = ig_cdf(hi[act], mm[act], lam) < qq[act]
    lo = np.minimum(seed, mm) * 0.5
    act = ig_cdf(lo, mm, lam) > qq
    for _ in range(80):
        if not act.any():
            break
        lo[act] *= 0.5
        act[act] = ig_cdf(lo[act], mm[act], lam) > qq[act]

    # Safeguarded Halley with active-set compression
    x = np.clip(seed, lo, hi)
    active = np.ones(x.shape, dtype=bool)
    for _ in range(maxiter):
        ai = active
        xs, qs, ms = x[ai], qq[ai], mm[ai]
        los, his = lo[ai], hi[ai]
        F = ig_cdf(xs, ms, lam) # noqa
        f = ig_pdf(xs, ms, lam)
        H = F - qs # noqa
        los = np.where(H < 0.0, xs, los)
        his = np.where(H > 0.0, xs, his)
        dlogf = _ig_logpdf_deriv_factor(xs, ms, lam)
        denom = 1.0 - (H / (2.0 * f)) * dlogf
        step = (H / f) / np.where(denom == 0.0, 1.0, denom)
        xn = xs - step
        bad = ~np.isfinite(xn) | (xn <= los) | (xn >= his)
        xn = np.where(bad, 0.5 * (los + his), xn)
        conv = np.abs(xn - xs) <= tol * np.abs(xn) + 1e-300
        x[ai] = xn
        lo[ai] = los
        hi[ai] = his
        # retire converged points from the active set
        idx = np.nonzero(ai)[0]
        active[idx[conv]] = False
        if not active.any():
            break

    out[work] = x
    return out.reshape(shape)


# Implied volatility -- the explicit solution.
def _solve_total_vol(q_otm, abs_k):
    """ v = sigma*sqrt(T) from the OTM-side probability argument q_otm. """
    mu = 2.0 / abs_k
    x = ig_ppf(q_otm, mu, lam=1.0)
    return 2.0 / np.sqrt(x)


def implied_vol_call(C, K, F, T): # noqa
    """ Black-Scholes implied volatility from a European call price.
        Args:
            C : call price (same units as D*F)
            K : strike
            F : forward
            T : time to maturity (years)
            D : risk-free discount factor (default 1.0)

            Returns sigma. Scalars in -> scalar out; arrays broadcast.

        Accuracy note
        -------------
        Recovery is machine-precision when the *out-of-the-money* normalized price exceeds ~1e-6.
        Inverting a deep in-the-money call (K << F) is inherently ill-conditioned: the price is
        dominated by intrinsic value and the tiny time-value carrying the vol information is lost
        to rounding. For best accuracy invert the OTM wing -- use this for K >= F and `implied_vol_put`
        for K <= F. Cells whose price sits at or beyond the no-arbitrage boundary return NaN rather
        than a misleading number.
    """
    C, K, F, T = map(lambda z: np.asarray(z, dtype=float), (C, K, F, T)) # noqa
    c = C / F
    k = np.log(K / F)
    atm = np.abs(k) < _ATM_TOL

    # m = 1 if K > F else K/F  (K<F branch maps ITM call to its OTM mirror)
    m = np.where(K > F, 1.0, K / F)
    q = (1.0 - c) / m
    return _assemble(q, c, k, atm, T)


def _assemble(q, c, k, atm, T): # noqa
    """Shared back end: map probability argument q to sigma, NaN if unrecoverable."""
    q, c, k, T = [np.asarray(z, dtype=float) for z in (q, c, k, T)] # noqa
    atm = np.asarray(atm, dtype=bool)
    q, c, k, atm, T = np.broadcast_arrays(q, c, k, atm, T) # noqa
    valid = (q > 0.0) & (q < 1.0)

    k_safe = np.where(atm | ~valid, 1.0, k)
    q_safe = np.where(valid, q, 0.5)
    v_ig = _solve_total_vol(q_safe, np.abs(k_safe))

    v_atm = 2.0 * ndtri(0.5 * (c + 1.0))
    v = np.where(atm, v_atm, v_ig)
    sigma = v / np.sqrt(T)
    sigma = np.where(valid | atm, sigma, np.nan)
    return sigma[()] if sigma.ndim == 0 else sigma


def implied_vol_put(P, K, F, T): # noqa
    """ Black-Scholes implied volatility from a European put price.
        Same conditioning note as `implied_vol_call`: best accuracy comes from the
        out-of-the-money wing (use this for K <= F).
    """
    P, K, F, T = map(lambda z: np.asarray(z, dtype=float), (P, K, F, T)) # noqa
    p = P / F
    k = np.log(K / F)
    atm = np.abs(k) < _ATM_TOL

    m = np.where(K > F, 1.0, K / F)
    q = (np.exp(k) - p) / m # probability argument for puts
    # ATM uses c = p (put-call parity at K=F gives c - p = 0)
    return _assemble(q, p, k, atm, T)


def implied_vol_schadner(expiry: npt.ArrayLike, strike: npt.ArrayLike, is_call: bool, fwd: npt.ArrayLike,
                         fwd_price: npt.ArrayLike, reroute_itm: bool=True) -> npt.NDArray[np.float64]:
    """ Black-Scholes inversion according to Schadner, using the inverse Gaussian.
        With optional rerouting of ITM.
    """
    if is_call:
        direct = implied_vol_call(fwd_price, strike, fwd, expiry)
        if reroute_itm:
            otm = (strike >= fwd)
            return np.where(otm, direct, implied_vol_put(fwd_price - fwd + strike, strike, fwd, expiry))
        else:
            return direct
    else:
        direct = implied_vol_put(fwd_price, strike, fwd, expiry)
        if reroute_itm:
            otm = (fwd >= strike)
            return np.where(otm, direct, implied_vol_call(fwd_price + fwd - strike, strike, fwd, expiry))
        else:
            return direct


if __name__ == "__main__":
    print("Hello")

