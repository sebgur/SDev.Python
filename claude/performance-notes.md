| `lru_cache` on `make_calendar` | 21.6 s | 2 lines |

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
