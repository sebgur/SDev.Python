""" Notes:
Container is a flat {(pair, date): report} dict — simplest to index into, and easy to later flatten into a DataFrame
 (one row per pair/date/tenor) if that's more convenient for analysis.
md_prov_factory is passed instead of a provider instance because Windows uses the spawn start method — objects handed
 to workers must pickle cleanly, and a class reference (or a functools.partial of one) does that safely whereas a live
  provider holding file handles/connections might not.
The if __name__ == "__main__": guard is required on Windows with spawn; _calibrate_one and calibrate_batch must stay at
 module level (not nested) so they're picklable.
Per-task try/except means one missing date (e.g. a holiday with no quotes) doesn't kill the whole batch — you'll see
 None in the results dict for that entry, and the error logged.
"""
import os
import datetime as dt
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from sdevpy.volatility.fx.fx_volcalib import FxVolCalibrator

log = logging.getLogger(__name__)


def _calibrate_one(pair: str, date: dt.date, md_prov_factory) -> dict:
    """ Runs in a worker process: build a fresh provider there rather than
        pickling a shared one across the process boundary. """
    md_prov = md_prov_factory()
    calibrator = FxVolCalibrator(pair, md_prov)
    return calibrator.calibrate(date)


def calibrate_batch(pairs: list[str], dates: list[dt.date], md_prov_factory,
                     max_workers: int = None) -> dict[tuple[str, dt.date], dict]:
    """ Calibrates every (pair, date) combination in parallel, one FxVolCalibrator
        surface per task. md_prov_factory is a zero-arg callable (e.g. a provider
        class, or functools.partial(MarketDataFileProvider, root=...)) — each
        worker calls it once for its own provider instance. """
    max_workers = max_workers or os.cpu_count()
    tasks = [(pair, date) for pair in pairs for date in dates]
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_calibrate_one, pair, date, md_prov_factory): (pair, date)
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
    from sdevpy.market.fileprovider import MarketDataFileProvider

    pairs = ["EURUSD", "USDJPY", "GBPUSD"]
    dates = [dt.datetime(2025, 12, d) for d in range(1, 20)]

    all_results = calibrate_batch(pairs, dates, MarketDataFileProvider)
    report = all_results[("EURUSD", dates[0])]   # {'tenor_reports': [...]}
