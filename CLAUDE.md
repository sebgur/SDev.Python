# Claude Code Instructions

## Python Code — Read-Only

**Never modify any Python files in this project.**

You may read, analyze, and explain Python code, but all edits are made exclusively by the user. When asked to fix a bug, add a feature, or refactor code, respond with suggestions only — describe what to change and why, but do not use the Edit or Write tools on any `.py` and `.ipynb` files.

## Memory
Do not read, write, or update any files in the memory directory. Do not use the auto-memory system.

## Code Review Rules

- Before reporting any issue, read the exact file and line number directly with the Read tool.
- Discard any finding that cannot be confirmed in the current file. Do not trust subagent summaries.
- Do not carry over findings from previous reviews without re-verifying them in the current working copy.

## Known and accepted — do not report

- Regression-style tests (sdevpy/tests/, e.g. test_mc.py, test_localvol_calib.py): pricing and
  calibration tests deliberately assert against stored reference values to detect changes, not
  against analytical or external benchmarks. Do not report the lack of benchmark/analytical
  tests or suggest replacing snapshots with benchmarks.
- MC time grid (sdevpy/montecarlo/mcpricer.py, build_timegrid): the simulation grid is
  deliberately independent of payoff event dates, to keep runtime bounded on large books.
  Event dates between grid nodes are linearly interpolated by design. Users are responsible
  for choosing n_timesteps fine enough for path-dependent payoffs.
  Examples/samples are not meant to be accurate. Do not report coarse-grid bias in these payoffs
  as a defect; bugs in the interpolation itself are still in scope.
- Local vol lookup (sdevpy/calibration/dataset.py): get_local_vols only retrieves calibrated
  models and never creates new ones; name_model_map intentionally holds the test underlyings
  (ABC, KLM, XYZ). Existing indices such as SPX are deliberately not supported.
- CI and tooling enforcement: there is deliberately no CI pipeline (.github/workflows), and
  ruff/mypy are run manually from time to time, not enforced. Do not report the absence of CI
  and do not treat the overall mypy error count as an issue. Still in scope: specific type
  annotations that are wrong or misleading (e.g. npt.ArrayLike on values that are always arrays,
  dict where a specific type is passed).