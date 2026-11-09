# import numpy as np
import pandas as pd
import datetime as dt
from sdevpy.conventions.repository import convention_repository
# from sdevpy.utilities import calendars as cdr
# from sdevpy.utilities import sessions as ssn


def swap_series_id(conv_id: str, expiry: str, tenor: str) -> str:
    """ Construct swap series ID from convention, expiry and tenor """
    series_id = conv_id + "^"
    if expiry is not None and expiry != "":
        series_id += expiry + "x"

    series_id += tenor
    return series_id


def swap_series(calib_repo, conv_ids: list[str], expiries: list[str], tenors: list[str],
                start_date: dt.datetime, end_date: dt.datetime|str|None, **kwargs) -> pd.DataFrame:
    """ Fetch par forward rate time series for several instruments
        Args:
            - conv_ids, expiries, tenors: same length, one series per tuple (conv_id, expiry, tenor)
            - start_date: datetime, None to start from earliest available
            - end_date: datetime, 'live' or None to end today. If it is the live trading day during a live
                        session, the last row is replaced with the live implied quotes
            - use_float_cache: price on a schedule built once per series (default True)
            - keep_missing: if True, keep dates where some series are missing (outer join), else only dates
                            common to all series (inner join)
        Returns a DataFrame with a 'Date' column and one column per series, named conv_id^expiry + x + tenor.
        Each cuve is loaded once and shared by all series that depend on it. """
    # use_float_cache = kwargs.get('use_float_cache', True)
    # keep_missing = kwargs.get('keep_missing', False)
    # scaler = 100.0 # ToDo
    conv_repo = convention_repository()

    # # Live end date: None, 'live' or today's trading day while in a live session
    # center = kwargs.get('center', None)
    # end_day, is_live = ssn.resolve_date_input(end_date, center=center, now=kwargs.get('now', None))
    # end_date = cdr.to_datetime(end_day)

    curves_by_id = {}
    # columns = {}
    # live_specs = {}
    for conv_id, expiry, tenor in zip(conv_ids, expiries, tenors, strict=True):
        # Series ID
        series_id = swap_series_id(conv_id, expiry, tenor)
        print(series_id)

        # Instrument details
        conv = conv_repo.swap(conv_id)
        index_id = conv.float_index

        # Curves
        curve_id = index_id
        if curve_id not in curves_by_id:
            available = calib_repo.get_available_curves([curve_id], start_date=start_date, end_date=end_date)
            if not available:
                raise ValueError(f"Curve {curve_id} not found between {start_date} and {end_date}")
            else:
                print(f"Available: {curve_id}")

    return pd.DataFrame({})


if __name__ == "__main__":
    from sdevpy.pricingcontext import default_calibration_repository

    calib_repo = default_calibration_repository()
    conv_ids = ['USD.SOFR.1D.SWAP', 'USD.SOFR.1D.SWAP']
    expiries = ['', '1Y']
    tenors = ['5Y', '5Y']
    start_date = dt.datetime(2025, 12, 15)
    end_date = None

    df = swap_series(calib_repo, conv_ids, expiries, tenors, start_date, end_date)
    print(df)
