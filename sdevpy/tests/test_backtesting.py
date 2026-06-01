import numpy as np
import pandas as pd
from sdevpy.timeseries.backtesting import backtest_one_trade


EXPECTED_KEYS = [
    '2D Realized Rtns from Trade Date in SD',
    '5D Realized Rtns from Trade Date in SD',
    '10D Realized Rtns from Trade Date in SD',
    '2D max draw down in SD',
    '5D max draw down in SD',
    '10D max draw down in SD',
    'basket stdev on Trade Date',
]


def _make_df(from_date, now_date):
    """ Linear drift + constant second asset. """
    dates = pd.bdate_range(from_date, now_date)
    n = len(dates)
    col1 = np.arange(n, dtype=float)
    col2 = np.ones(n, dtype=float)
    df = pd.DataFrame({'A': col1, 'B': col2}, index=dates.strftime('%Y-%m-%d'))
    return df


def test_return_keys():
    from_date, trade_date, now_date = '2022-01-03', '2022-03-01', '2022-06-01'
    df = _make_df(from_date, now_date)
    result = backtest_one_trade(from_date, trade_date, now_date, df, [1.0, 0.0], -1.0)
    assert set(result.keys()) == set(EXPECTED_KEYS)


def test_zscore_negative_buys_sign():
    from_date, trade_date, now_date = '2022-01-03', '2022-03-01', '2022-06-01'
    df = _make_df(from_date, now_date)
    r_buy = backtest_one_trade(from_date, trade_date, now_date, df, [1.0, 0.0], -1.0)
    r_sell = backtest_one_trade(from_date, trade_date, now_date, df, [1.0, 0.0], +1.0)
    for key in ['2D Realized Rtns from Trade Date in SD',
                '5D Realized Rtns from Trade Date in SD',
                '10D Realized Rtns from Trade Date in SD']:
        assert np.isclose(r_buy[key], -r_sell[key])


def test_draw_down_nonpositive():
    from_date, trade_date, now_date = '2022-01-03', '2022-03-01', '2022-06-01'
    df = _make_df(from_date, now_date)
    for zscore in [-1.0, +1.0]:
        result = backtest_one_trade(from_date, trade_date, now_date, df, [1.0, 0.0], zscore)
        for key in ['2D max draw down in SD', '5D max draw down in SD', '10D max draw down in SD']:
            assert result[key] <= 0.0


def test_stdev_is_positive():
    from_date, trade_date, now_date = '2022-01-03', '2022-03-01', '2022-06-01'
    df = _make_df(from_date, now_date)
    result = backtest_one_trade(from_date, trade_date, now_date, df, [1.0, 0.0], -1.0)
    assert result['basket stdev on Trade Date'] > 0.0
