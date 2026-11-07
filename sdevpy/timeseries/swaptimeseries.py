import numpy as np
import pandas as pd
import datetime as dt
from sdevpy.conventions.repository import convention_repository
from sdevpy.utilities import calendars as cdr
from sdevpy.utilities import sessions as ssn


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
    # conv_repo = convention_repository()

    # Live end date: None, 'live' or today's trading day while in a live session
    center = kwargs.get('center', None)
    end_day, is_live = ssn.resolve_end_date(end_date, center=center, now=kwargs.get('now', None))
    end_date = cdr.to_datetime(end_day)

    return pd.DataFrame({})
