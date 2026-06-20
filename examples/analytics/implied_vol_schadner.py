import numpy as np
import time
from scipy.special import ndtri
from sdevpy.analytics import black
from sdevpy.analytics import schadner


# Self-test: reproduce the paper's recovery grid (Section 3, Eqs. 3 & 13).
def test_schadner():
    method = 'schadner'
    # method = 'newton'

    # Set vols and deltas
    # vol_grid = np.asarray([0.01, 0.05, 0.20])
    # delta_grid = np.array([0.30, 0.45, 0.55, 0.70])
    vol_grid = np.concatenate(([0.01], np.arange(0.05, 2.0001, 0.05)))
    delta_grid = np.array([0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95])
    print(f"vol_grid: {vol_grid.shape}")
    print(f"delta_grid: {delta_grid.shape}")

    vol, delta = np.meshgrid(vol_grid, delta_grid, indexing="ij")
    vol = vol.ravel()
    delta = delta.ravel()
    print(f"Grid points: {vol.size}")

    # Set strikes from deltas
    t = 1.5
    m = vol * (0.5 * vol - ndtri(delta))
    fwd = 100.0
    k = fwd * np.exp(m)

    # Round-trip on call price
    call = black.price(t, k, True, fwd, vol)
    match method:
        case 'schadner':
            sigma_call = schadner.implied_vol_schadner(t, k, True, fwd, call)
            # sigma_call = implied_vol_call(call, k, fwd, t)
        case 'newton':
            sigma_call = black.implied_vol_newton(t, k, True, fwd, call)
        case 'brent':
            raise ValueError('not yet')
        case _:
            raise ValueError(f"Unknown method: {method}")

    err_call = np.abs(sigma_call - vol) # recovered total vol vs input
    print(f"Mean abs recovery error(call): {err_call.mean():.3e}")
    print(f"Max abs recovery error(call): {err_call.max():.3e}")
    print("(paper reports mean 2.24e-16, max 1.33e-15)")

    # Rought-trip on put price
    put = call - (fwd - k)
    match method:
        case 'schadner':
            sigma_put = schadner.implied_vol_schadner(t, k, False, fwd, put)
            # sigma_put = implied_vol_put(put, k, fwd, t)
        case 'newton':
            sigma_put = black.implied_vol_newton(t, k, False, fwd, call)
        case 'brent':
            raise ValueError('not yet')
        case _:
            raise ValueError(f"Unknown method: {method}")
    err_put = np.abs(sigma_put - vol)
    print(f"Mean abs recovery error(put): {err_put.mean():.3e}")
    print(f"Max abs put recovery err: {err_put.max():.3e}")

    # Performance (vectorised)
    reps = 1000
    t0 = time.perf_counter()
    for _ in range(reps):
        match method:
            case 'schadner':
                schadner.implied_vol_call(call, k, fwd, t)
            case 'newton':
                black.implied_vol_newton(t, k, False, fwd, call)
            case 'brent':
                raise ValueError('not yet')
            case _:
                raise ValueError(f"Unknown method: {method}")
    dt = time.perf_counter() - t0
    per = dt / (reps * vol.size) * 1e6
    print(f"Vectorised speed: {per:.3f} us/eval over {reps*vol.size:,} evals")

    # Well-conditioned random batch: invert the OTM wing, prices not tiny
    rng = np.random.default_rng(1)
    n = 200_000
    fr = 100 * np.exp(rng.normal(0, 0.2, n))
    kr = fr * np.exp(rng.normal(0, 0.3, n))
    tr = rng.uniform(0.05, 3.0, n)
    sr = rng.uniform(0.05, 1.2, n)
    callr = black.price(tr, kr, True, fr, sr)
    putr = callr - (fr - kr)

    # Route to the OTM wing and keep only recoverable prices (norm. price > 1e-6)
    otm_call = (kr >= fr)
    rec = schadner.implied_vol_call(callr, kr, fr, tr)
    rec = np.where(otm_call, rec, schadner.implied_vol_put(putr, kr, fr, tr))
    otm_price = np.where(otm_call, callr, callr - (fr - kr))
    keep = (otm_price > 1e-6 * fr)
    er = np.abs(rec[keep] - sr[keep])
    print(f"Random OTM batch: n={keep.sum():,}  "
          f"Mean={er.mean():.2e} max={er.max():.2e}")

    # Graceful failure: price at the no-arbitrage boundary -> NaN, not 0
    nan_demo = schadner.implied_vol_call(C=1e-300, K=400.0, F=100.0, T=0.1)
    print(f"Deep-OTM underflow price -> {nan_demo} (NaN = unrecoverable)")


if __name__ == "__main__":
    test_schadner()
