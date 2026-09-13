# FX vol calibration — performance findings

Profiling session on `sdevpy/volatility/fx/fx_volcalib.py`, 2026-09-13.
Baseline: `FxVolCalibrator('USDJPY').calibrate(2025-12-15)`, 6 tenors × 2 quoted deltas
(+ 2 extra tail deltas), ≈ 20–26 s wall clock (varies a few seconds run to run).

**Every change below was verified to produce bit-identical strikes and vols
(max abs difference 0.000e+00 against the unoptimised baseline).** None of them
trade accuracy for speed.

## Summary

| Change | Cumulative | Effort |
|---|---|---|
| baseline | 26.4 s | — |
| `lru_cache` on `make_calendar` | 21.6 s | 2 lines |
| + `scipy.special.ndtr` instead of `scipy.stats.norm.cdf` | 8.5 s | ~10 lines, 3 files |
| + branch skipping and `.max()` in `strike_from_delta` | 4.4 s | ~20 lines, 1 file |
| + `ndtr` in `black.py` / `fx_vannavolga.py` | 3.4 s | ~5 lines |

**7.7x total.** Full vectorisation (see last section) could plausibly reach 15–20x
**in total, not on top of the 7.7x** — the two do not multiply out to 100x+, see
"Why the gains do not fully compound" below. It is days of work rather than hours.

## What is NOT the bottleneck

The original hypothesis was the repeated `datetime` → year-fraction conversion inside
`strike_from_delta`. **It is not significant.** `fx_market_yearfraction` does not appear
in the top 28 entries by cumulative time; it is called once per `strike_from_delta`
invocation (401 times in a full run), which is noise. The cost is in what happens
*after* it, in the solver loops.

## 1. `scipy.stats.norm.cdf` → `scipy.special.ndtr` (worth ~3x alone)

`norm.cdf` delegates to `ndtr` internally (`norm_gen._cdf` → `_norm_cdf` → `sp.ndtr`),
so results are **bitwise identical**. The difference is a fixed ~60 µs per-call wrapper
overhead (argument validation, `argsreduce`, loc/scale broadcasting) that is independent
of array size:

```
  array size     norm.cdf         ndtr    ratio
           1       58.8us        0.4us   136.8x
         100       61.0us        1.2us    52.1x
        1000       92.1us        8.8us    10.4x
       10000      391.6us      124.4us     3.1x
     1000000    43526.2us    15310.3us     2.8x
```

The calibrator calls it ~141,000 times on 1-element arrays inside bisection loops —
the worst case for this overhead. `argsreduce` alone cost 6.1 s of the 32 s profile.

`ndtr` is a genuine `numpy.ufunc` (broadcasting, `out=`, `where=` all work), so
**no vectorisation is lost**. Precedent already exists in this repo:
`sdevpy/analytics/schadner.py:28` imports from `scipy.special` directly.

### Exact changes

**`sdevpy/volatility/fx/fx_deltastrike.py`**

```python
# line 38
from scipy.special import ndtr, ndtri

# lines 106-107, in bs_delta
    raw = phi * ndtr(phi * d1)
    pa = phi * (k / f) * ndtr(phi * d2)

# line 237, branch 1 closed-form inversion
        d1_cf = phi * ndtri(x_cf)

# line 244, _put_pa_delta
        return -disc * (k / f) * ndtr(-d2)

# line 266, _call_pa_delta
        return disc * (k / f) * ndtr(d2)

# line 331, __main__ demo
    print("check delta:", -ndtr(-d1) * np.exp(-0.02 * 0.5))
```

**`sdevpy/analytics/black.py`**

```python
# line 5 (keep the minimize_scalar import on line 6)
from scipy.special import ndtr

# near the top
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

# line 19, in price
    return w * (fwd * ndtr(w * d1) - strike * ndtr(w * d2))

# line 85, in the implied-vol Newton loop (norm.pdf has no scipy.special equivalent)
        vega = fwd * (_INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)) * sqrt_t
```

**`sdevpy/volatility/fx/fx_vannavolga.py`**

```python
# line 28: delete "from scipy.stats import norm" (pdf was its only use)
# line 45, in bs_vega
    return fwd * (_INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)) * sqrt_t
```

Consider putting a shared `norm_pdf(x)` helper in `black.py` rather than repeating
the constant — `fx_vannavolga.py` already imports from `sdevpy.analytics.black`.

## 2. `strike_from_delta` computes all three branches unconditionally

At `fx_deltastrike.py:233-291` the closed-form branch, the premium-adjusted put
bisection, and the premium-adjusted call ternary search + two bisections **all run on
every call**, with `np.where` picking the winner at the end. But every call in this
calibrator solves exactly one scalar put *or* one scalar call.

Evidence: 401 calls produced 96,428 `_call_pa_delta` evaluations (240 each) and
37,668 `_put_pa_delta` (94 each). Roughly half is discarded.

Fix — compute the masks up front and guard each branch:

```python
    is_pa   = prem_adjusted
    is_put  = phi < 0
    is_call = ~is_put
    need_cf      = bool(np.any(~is_pa))
    need_pa_put  = bool(np.any(is_pa & is_put))
    need_pa_call = bool(np.any(is_pa & is_call))
```

then wrap branch 1 in `if need_cf:`, branch 2 in `if need_pa_put:`, branch 3 in
`if need_pa_call:`, with `nan_like = np.full(s.shape, np.nan)` (and
`valid_cf = np.zeros(s.shape, dtype=bool)`, `n_sol_call = np.zeros(s.shape, dtype=int)`)
as the else-branch placeholders. The combining logic at the end is unchanged.

This cut `_call_pa_delta` calls from 96,428 to 47,366. Stays correct for genuinely
mixed vectorised grids — when all branches are needed, the guards are three cheap
boolean reductions.

## 3. `np.nanmax` in the solver convergence checks (~26% of post-ndtr runtime)

`fx_deltastrike.py:133` and `:152` call `np.nanmax(hi - lo)` once per bisection /
ternary iteration — 87,051 times, 3.4 s of a 13 s profile. `np.nanmax` is 6.2x slower
than `ndarray.max()` on small arrays (7.18 µs vs 1.15 µs) because `_replace_nan`
copies the array first. The brackets are log-strike bounds and cannot be NaN:

```python
        if (hi - lo).max() < tol:      # was np.nanmax(hi - lo)
```

## 4. `make_calendar` rebuilds 101 years of holidays on every call

128 ms per call, ~20 calls per `calibrate()` (≈2.5 s). `fx_option_dates` calls both
`fx_pillar_date` and `fx_spot_date`, and each builds its calendar from scratch
(`fxforward.py:94`, `:114`), looping years 2000–2100 through the `holidays` registry —
4,040 holiday-object constructions per run.

```python
# sdevpy/utilities/scalendar.py:359
@functools.lru_cache(maxsize=None)
def make_calendar(name: str, start_year: int=2000, end_year: int=2100):
```

`Calendar` holds an immutable holiday set, so caching is safe. Only worth ~1.2x on a
6-tenor surface but scales with surface size, and it is a two-line change.

## 5. Full vectorisation — feasible, bigger job

Grid solves are almost free: all elements march through the same ~350 solver
iterations in lockstep, so cost is nearly independent of grid size.

```
  problems    1 grid call    per problem
         1         44.4ms      44440.1us
        24         47.3ms       1972.3us
       120         51.4ms        428.0us
      2400        129.8ms         54.1us
```

Solving 24 problems costs 7% more than solving one. A measured batch of the 24 pillar
strikes (one `(6,4)` grid call vs 24 scalar calls) gave **22x**, agreeing to 6.5e-11.

Where the 401 calls come from:

```
   190  (47.4%)  _smile_from_smile_butterfly   <- inside the Brent objective
   163  (40.6%)  _vol_at_delta                 <- tail fixed-point
    24  ( 6.0%)  market_strangle               <- MS target strikes
    24  ( 6.0%)  calibrate_tenor               <- pillar strikes
```

Three tiers of difficulty:

- **48 easy calls (12%)** — pillar and MS target strikes. Fully independent across
  (tenor, delta, wing); collapse to 2 grid calls with no algorithm change. Small
  end-to-end effect because it is only 12% of calls.
- **163 tail calls (41%)** — the `sigma → strike → vol → sigma` fixed point. Same
  iteration elementwise, so run all (tenor, tail-delta, wing) in lockstep with a
  convergence mask (the pattern `_vectorized_bisect` already uses). 163 → ~7 calls.
  Blocker: `VannaVolgaSmile` holds scalar floats, one object per tenor. The maths
  inside (`lagrange_weights`, `vv_weights`, `price`, `_exact_vol`) is all NumPy and
  would broadcast unchanged if those fields became arrays — contained, but a real
  design change to a single-expiry dataclass.
- **190 Brent calls (47%)** — the ceiling. `calibrate_smile_strangle` uses
  `scipy.optimize.brentq`, which is irreducibly scalar: 12 independent smile-strangle
  problems mean 12 sequential solves, ~6 `build_smile` evaluations each. Batching needs
  a vectorised root finder (template: `_vectorized_bisect`). Brent converges in ~6
  evaluations where bisection needs 40–50, so 190 → ~45 grid calls — a 4x call
  reduction, the least favourable ratio. A vectorised Newton would do better since the
  objective is smooth and near-linear in the butterfly, but is more work to make robust.

**Estimate:** 401 calls → ~54 grid calls ≈ 2.5 s, and with `ndtr` on top, around 1 s.
So roughly **15–20x in total** — this is *not* multiplied onto the 7.7x.

### Why the gains do not fully compound

Measured on the fully-optimised (7.7x) state:

```
fully-optimised run: 3.89 s
  inside strike_from_delta : 2.50 s  (64%)
  everything else          : 1.39 s  (36%)

Amdahl ceiling if strike_from_delta went to ZERO: 2.8x further
```

Batching only attacks `strike_from_delta`. Once the cheap fixes have made each call
~7.7x cheaper, that function is only 64% of what remains, so batching is capped at
2.8x further even in the impossible limit — realistically 2–2.5x, since 54 grid calls
still cost something. Hence 7.7 × ~2.5 ≈ **19x total**.

The fixes do compound in principle (they reduce different factors: cost per iteration,
iterations per call, number of calls), but each shrinks the slice the next can work on.
Applied in the opposite order the total is the same, just attributed differently.

The 36% that batching cannot touch is the next bottleneck: mostly the Schadner
implied-vol inversion inside `VannaVolgaSmile._exact_vol`, smile construction, and
date arithmetic.

### Keep the wings in separate grid calls

Branch skipping (§2) mostly stops working if puts and calls share one grid — both
`need_pa_put` and `need_pa_call` are then true, so only the cheap closed-form branch
is skipped:

```
one MIXED grid (24 problems), original        : 50.3 ms
one MIXED grid (24 problems), branch-skip     : 16.6 ms   (3.02x, mostly the .max() fix)
TWO SEPARATE grids (puts | calls), branch-skip : 11.9 ms   (4.22x)
```

When vectorising, batch one grid of puts and one grid of calls so each skips an
expensive branch.

The stronger argument is scaling, not today's runtime: the scalar design grows linearly
with tenors × deltas, the batched design is nearly flat out to thousands of problems.
A production surface of 15 tenors × 5 deltas would roughly quadruple current runtime
and barely move a batched implementation.

## How to reproduce

```python
import datetime as dt, cProfile, pstats, sys, os
import sdevpy.volatility.fx.fx_volcalib as fvc
from sdevpy.market.fileprovider import MarketDataFileProvider

c = fvc.FxVolCalibrator('USDJPY', MarketDataFileProvider())
sys.stdout = open(os.devnull, 'w')          # the method prints per tenor
pr = cProfile.Profile(); pr.enable()
c.calibrate(dt.datetime(2025, 12, 15))
pr.disable(); sys.stdout = sys.__stdout__
pstats.Stats(pr).sort_stats('tottime').print_stats(20)
```

To verify a change is lossless, capture `report['tenor_reports']` before and after and
compare `strikes` / `vols` elementwise — every change above gives exactly 0.0.
