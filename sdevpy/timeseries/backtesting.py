import os
import numpy as np
import pandas as pd
import datetime as dt
import math
from sdevpy.timeseries import timeseriestools as tst

from sdevpy.cointegration import data_io as myio
from sdevpy.cointegration import back_testing as btest
from sdevpy.cointegration import utils as ut


def backtest_one_trade(FROM, TRADE_DATE, NOW, df_data, weights, zscore_TRADE_DATE):
    """ Given a fixed basket, from date, trade date, now date, compute the sharpe ratio, z_score,
        5D realized return and 10D realized return from trade date """
    # Compute the basket up to NOW, because we need the realized 5D and 10D returns
    series_up_to_NOW = tst.weighted_series(df_data.loc[FROM:NOW], weights)

    # Compute the stdev as if we were on the TRADE DATE, this is used for computing the return in terms of ZScore
    stdev_to_tdate = np.std(series_up_to_NOW.loc[:TRADE_DATE])
    series_tdate_to_NOW = series_up_to_NOW.loc[TRADE_DATE:NOW]

    returns2D_from_tdate = tst.x_day_historical_returns_in_SD(series_tdate_to_NOW, 2, stdev_to_tdate).iloc[0]
    returns5D_from_tdate = tst.x_day_historical_returns_in_SD(series_tdate_to_NOW, 5, stdev_to_tdate).iloc[0]
    returns10D_from_tdate = tst.x_day_historical_returns_in_SD(series_tdate_to_NOW, 10, stdev_to_tdate).iloc[0]

    min_max_res_2D = tst.min_max_return_x_days_in_SD(series_tdate_to_NOW, 2, stdev_to_tdate).iloc[0]
    min_2D_rtns_in_SD = min_max_res_2D['min']
    max_2D_rtns_in_SD = min_max_res_2D['max']

    min_max_res_5D = tst.min_max_return_x_days_in_SD(series_tdate_to_NOW, 5, stdev_to_tdate).iloc[0]
    min_5D_rtns_in_SD = min_max_res_5D['min']
    max_5D_rtns_in_SD = min_max_res_5D['max']

    min_max_res_10D = tst.min_max_return_x_days_in_SD(series_tdate_to_NOW, 10, stdev_to_tdate).iloc[0]
    min_10D_rtns_in_SD = min_max_res_10D['min']
    max_10D_rtns_in_SD = min_max_res_10D['max']

    if math.isnan(returns10D_from_tdate):
        raise Exception('Trade DATE is less than 10 days ago.')

    if zscore_TRADE_DATE > 0:
        # Flip the return sign if zscore is above 0, because we sell. The basket goes down and we earn
        returns2D_from_tdate = -returns2D_from_tdate
        returns5D_from_tdate = -returns5D_from_tdate
        returns10D_from_tdate = -returns10D_from_tdate

        # So the max draw down is the negative of the max returns
        max_2D_draw_down_in_SD = -max_2D_rtns_in_SD
        max_5D_draw_down_in_SD = -max_5D_rtns_in_SD
        max_10D_draw_down_in_SD = -max_10D_rtns_in_SD
    else:
        # We buy so that max draw down is teh min returns
        max_2D_draw_down_in_SD = min_2D_rtns_in_SD
        max_5D_draw_down_in_SD = min_5D_rtns_in_SD
        max_10D_draw_down_in_SD = min_10D_rtns_in_SD

    # If the max draw down is a positive number, we floor it to 0 to show there is no loss
    max_2D_draw_down_in_SD = np.minimum(max_2D_draw_down_in_SD, 0.0)
    max_5D_draw_down_in_SD = np.minimum(max_5D_draw_down_in_SD, 0.0)
    max_10D_draw_down_in_SD = np.minimum(max_10D_draw_down_in_SD, 0.0)

    return {'2D Realized Rtns from Trade Date in SD': returns2D_from_tdate,
            '5D Realized Rtns from Trade Date in SD': returns5D_from_tdate,
            '10D Realized Rtns from Trade Date in SD': returns10D_from_tdate,
            '2D max draw down in SD': max_2D_draw_down_in_SD,
            '5D max draw down in SD': max_5D_draw_down_in_SD,
            '10D max draw down in SD': max_10D_draw_down_in_SD,
            'basket stdev on Trade Date': stdev_to_tdate}


if __name__ == "__main__":
    ROOT = r"C:\\temp\\sdevpy\\cointegration"
    FROM = '2013-04-23'
    TRADE_DATE = '2020-01-15'
    NOW = '2020-06-02'
    tickers = ['AUDUSD Curncy', 'NZDUSD Curncy', 'CADUSD Curncy']
    weights = [1.0, -0.77030425, -0.41245188] 
    zscore_TRADE_DATE = -1.508

    data_file = os.path.join(ROOT, "fx_spots.tsv")
    df_raw = pd.read_csv(data_file, sep='\t')
    df_raw.set_index('Dates', inplace=True)
    df_fx_spots = df_raw.loc[FROM:NOW]
    df_data = df_fx_spots[tickers]
    backtest = backtest_one_trade(FROM, TRADE_DATE, NOW, df_data, weights, zscore_TRADE_DATE)
    print(backtest)

    print(round(10000.0 * (backtest['5D Realized Rtns from Trade Date in SD']- -0.0939747144428266), 4))
    print(round(10000.0 * (backtest['5D max draw down in SD'] - -0.19568917067110778), 4))
    print(round(10000.0 * (backtest['basket stdev on Trade Date'] - 0.017399315329440185), 4))
