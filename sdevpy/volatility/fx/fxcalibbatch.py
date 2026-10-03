""" Notes:
Container is a flat {(pair, date): report} dict — simplest to index into, and easy to later flatten into a DataFrame
 (one row per pair/date/tenor) if that's more convenient for analysis.
md_repo_factory is passed instead of a repository instance because Windows uses the spawn start method — objects handed
 to workers must pickle cleanly, and a class reference (or a functools.partial of one) does that safely whereas a live
  repository holding file handles/connections might not.
The if __name__ == "__main__": guard is required on Windows with spawn; _calibrate_one and calibrate_batch must stay at
 module level (not nested) so they're picklable.
Per-task try/except means one missing date (e.g. a holiday with no quotes) doesn't kill the whole batch — you'll see
 None in the results dict for that entry, and the error logged.
"""
import os
import datetime as dt
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from sdevpy.volatility.fx.fxvolcalib import FxVolCalibrator
log = logging.getLogger(__name__)


def _calibrate_one(pair: str, date: dt.date, md_repo_factory, cal_repo_factory) -> dict:
    """ Runs in a worker process: build a fresh provider there rather than
        pickling a shared one across the process boundary. """
    md_repo = md_repo_factory()
    cal_repo = cal_repo_factory()
    calibrator = FxVolCalibrator(pair, md_repo, cal_repo)
    return calibrator.calibrate(date)


def calibrate_batch(pairs: list[str], dates: list[dt.date], md_repo_factory, cal_repo_factory,
                     max_workers: int = None) -> dict[tuple[str, dt.date], dict]:
    """ Calibrates every (pair, date) combination in parallel, one FxVolCalibrator
        surface per task. md_repo_factory and cal_repo_factory are zero-arg callable (e.g. a provider
        class, or functools.partial(default_market_repository, default_calibration_repository, root=...)) — each
        worker calls it once for its own provider instance. """
    max_workers = max_workers or os.cpu_count()
    tasks = [(pair, date) for pair in pairs for date in dates]
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_calibrate_one, pair, date, md_repo_factory, cal_repo_factory): (pair, date)
                   for pair, date in tasks}
        for future in as_completed(futures):
            pair, date = futures[future]
            try:
                results[(pair, date)] = future.result()
            except Exception as exc:
                log.error(f"Calibration failed for {pair} on {date}: {exc}")
                results[(pair, date)] = None

    return results


if __name__ == "__main__":
    from sdevpy.pricingcontext import default_market_repository, default_calibration_repository

    pairs = ["EURUSD", "USDJPY", "GBPUSD"]
    dates = [dt.datetime(2025, 12, d) for d in range(1, 20)]

    all_results = calibrate_batch(pairs, dates, default_market_repository, default_calibration_repository)
    report = all_results[("EURUSD", dates[0])]   # {'tenor_reports': [...]}
