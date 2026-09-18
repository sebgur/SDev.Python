| `lru_cache` on `make_calendar` | 21.6 s | 2 lines |
| + branch skipping and `.max()` in `strike_from_delta` | 4.4 s | ~20 lines, 1 file |

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
