import os
import datetime as dt
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from ta.momentum import RSIIndicator
from sdevpy.timeseries import meanreversion as mr
from sdevpy.timeseries import timeseriestools as tst

from sdevpy.cointegration import data_io as myio
from sdevpy.cointegration import coint_trading as ct
from sdevpy.tools import clipboard as clipboard

# [johansen_test_diag]
# Looks like we're going to compute the time series of the linear product and then fit
# a mean reversion to it. Then this no longer has any relation to the Johansen test.
# So wouldn't it be better to make this function more generic and focused on/named after
# what it does with the mean reversion?
def johansen_diagnostics(test, df_data, verbose=True, det_order=0, k_ar_diff=1):
    """ Compute the diagnostics of a basket fom Johansen's test results """
    bool_trace_5pct = test['trace (5%)']
    bool_trace_10pct = test['trace (10%)']
    bool_eigen_5pct = test['eigen (5%)']
    bool_eigen_10pct = test['eigen (10%)']
    weights = test['weights']

    # Compute the Johansen dot product series with the estimated weights
    johansen_basket = pd.Series(np.dot(df_data, weights), name='Series')
    johansen_basket.index = df_data.index

    # Compute the mean reversion time series statistics
    mr_ts = mr.MeanRevertingTimeSeries(johansen_basket)
    mr_level = mr_ts.get_mr_level()
    mr_rate = mr_ts.get_mr_rate()
    time_in_days = mr_ts.get_half_life()
    current_level = mr_ts.get_current_level()
    daily_hist_normal_vol = tst.last_daily_hist_normal_vol(johansen_basket)
    current_zscore = mr_ts.get_current_zscore()

    res_sharpe = mr.compute_sharpe_ratio(mr_level, mr_rate, time_in_days, current_level,
                                         daily_hist_normal_vol, current_zscore)

    sharpe_ratio_half_life = res_sharpe['Sharpe Ratio']

    half_life = mr_ts.get_half_life()
    series_stdev = mr_ts.get_stdev()

    position_df, USD_amount = tst.create_position(df_data, weights)

    what_you_should_trade = list(position_df['market convention notional'].values)
    what_you_should_trade_USD_amount = USD_amount

    rsi_14 = RSIIndicator(johansen_basket, 14).rsi().iloc[-1]

    # Flip the sign of the basket if SD is +ve, to indicate that we sell the basket
    if current_zscore > 0:
        for i in range(len(what_you_should_trade)):
            what_you_should_trade[i] = -what_you_should_trade[i]

        what_you_should_trade_USD_amount = -1 * what_you_should_trade_USD_amount

    if verbose:
        const_pvalue = round(mr_ts.get_const_pvalue(), 8)
        series_pvalue = round(mr_ts.get_series_pvalue(), 8)
        print(' -------------------------------------------------------')
        print(position_df)
        print('USD_amount = ' + str(round(USD_amount, 4)))
        print(' -------------------------------------------------------')
        print('Pass trace test (5%) = ' + str(bool_trace_5pct))
        print('Pass eigen test (5%) = ' + str(bool_eigen_5pct))
        print('Pass trace test (10%) = ' + str(bool_trace_10pct))
        print('Pass eigen test (10%) = ' + str(bool_eigen_10pct))
        series_johansen = coint_johansen(df_data, det_order, k_ar_diff)
        print(trace_stats(series_johansen))
        print(eigen_stats(series_johansen))
        print(' -------------------------------------------------------')
        print('Half life = ' + str(round(half_life, 3)))
        print('Const p-value = ' + str(const_pvalue))
        print('Series p-value = ' + str(series_pvalue))
        print('Current zscore = ' + str(round(current_zscore, 3)))
        print('SD Sharpe Ratio = ' + str(round(sharpe_ratio_half_life, 3)))
        print('RSI 14 = ' + str(round(rsi_14, 3)))
        print('Current level = ' + str(round(current_level, 3)))
        print('1mio 1SD in USD = ' + str(round(1e6 * series_stdev, 0)))
        print('Series stdev = ' + str(series_stdev))
        print('MR level = ' + str(mr_level))
        print('MR rate = ' + str(mr_ts.get_mr_rate()))

    rounded_weights = [round(w, 8) for w in weights]

    # Round PX_LAST and what_you_should_trade to 4 decimal for easy printing
    PX_LAST = list(position_df['PX_LAST'].values) 
    what_you_should_trade = [round(w, 4) for w in what_you_should_trade]
    PX_LAST = [round(w, 4) for w in PX_LAST]

    return {'Half life': half_life, 'Rounded weights': rounded_weights,
            'What you should trade': what_you_should_trade,
            'What you should trade USD amount': what_you_should_trade_USD_amount,
            'Current zscore': current_zscore,
            'Trace (5%)': bool_trace_5pct, 'Eigen (5%)': bool_eigen_5pct,
            'Trace (10%)': bool_trace_10pct, 'Eigen (10%)': bool_eigen_10pct,
            'Johansen Series': johansen_basket, '1mio 1SD in USD': 1e6 * series_stdev,
            'PX_LAST' : PX_LAST, 'Series stdev': series_stdev, 'MR level': mr_level,
            'Half life Sharpe Ratio': sharpe_ratio_half_life, 'RSI 14': rsi_14}

# [johansen_test_estimation]
def johansen_test(df_data, det_order = 0, k_ar_diff = 1):
    """ Estimate the weights and test statistics """
    # Run Johansen test 
    res_jo = coint_johansen(df_data, det_order, k_ar_diff)

    # Check the trace and eigenvalue test of Johansen 
    trace_5pct, trace_10pct, eigen_5pct, eigen_10pct = check_johansen_stats_fast(res_jo)

    # Get the normalized lst eigenvector which is the weights 
    weights = norm_1st_eigvec(res_jo)

    return {'weights': weights, 'trace (5%)': trace_5pct, 'eigen (5%)': eigen_5pct, 
            'trace (10%)': trace_10pct, 'eigen (10%)': eigen_10pct}

# [check_johansen_test_stats_fast]
def check_johansen_stats_fast(res_jo):
    """ Extract test stats for trace/eigen at 5% and 10% confident intervals """

    # Trace_stats
    trace_test_stats = res_jo.lr1[0]
    trace_10_pct = res_jo.cvt[0][0]
    trace_5_pct = res_jo.cvt[0][1]
    #trace_1_pct = res_jo.cvt[0][2]

    bool_trace_5pct = False
    if trace_test_stats > trace_5_pct:
        bool_trace_5pct = True

    bool_trace_10pct = False 
    if trace_test_stats > trace_10_pct:
        bool_trace_10pct = True

   # Eigen stats
    eigen_test_stats = res_jo.lr2[0]
    eigen_10_pct = res_jo.cvm[0][0]
    eigen_5_pct = res_jo.cvm[0][1]
    #eigen_1_pct = res_jo.cvm[0][2]

    bool_eigen_5pct = False
    if eigen_test_stats > eigen_5_pct:
        bool_eigen_5pct = True

    bool_eigen_10pct = False
    if eigen_test_stats > eigen_10_pct:
        bool_eigen_10pct = True

    return bool_trace_5pct, bool_trace_10pct, bool_eigen_5pct, bool_eigen_10pct


def trace_stats(res_jo):
    """ Retrieve the trace test statistics from the result of Johansen test in a pretty format """
    # Find out the size of the vector. This is the number of assets.
    N = len(res_jo.lr1)

    data = np.zeros((N, 4))

    # create the row labels
    index_list = ['r=0']
    for i in range(1, N):
        index_list.append('r<=' + str(i))

    # populate the data in the desired format
    for i in range(N):
        data[i][0] = res_jo.lr1[i]
        for j in range(3):
            data[i][j+1] = res_jo.cvt[i][j]

    res_df = pd.DataFrame(data, columns = ['trace', '10%', '5%', '1%'], index = index_list)
    return res_df


def eigen_stats(res_jo):
    """ Retrieve the eigen test statistics from the result of Johansen test in a pretty format """
    # Find out the size of the vector. This is the number of assets.
    N = len(res_jo.lr2)

    data = np.zeros((N, 4))

    # create the row labels
    index_list = ['r=0']
    for i in range(1, N):
        index_list.append('r<=' + str(i))

    # populate the data in the desired format
    for i in range(N):
        data[i][0] = res_jo.lr2[i]
        for j in range(3):
            data[i][j+1] = res_jo.cvm[i][j]

    res_df = pd.DataFrame(data, columns = ['eigen', '10%', '5%', '1%'], index = index_list)
    return res_df


def norm_1st_eigvec(res_jo):
    """ Retrieve the normalized first eigen vector from the result of Johansen test.
        The normalized vector are the weights of the cointegrated basket """
    # Size of the eigenvector
    N = len(res_jo.evec)
    first_eigvec = []
    for i in range(N):
        first_eigvec.append(res_jo.evec[i][0])

    # Normalized with respect to the first element
    return np.array(first_eigvec) / res_jo.evec[0][0]


def convert_to_currency(df, target_ccy):
    """ Convert FX spot data to target currency """
    converted_df = df
    converted_df.set_index('Dates')
    for col in converted_df.columns[1:]:
        is_usd_for, inverse_ticker = is_fx_for_ticker(col, target_ccy)
        if is_usd_for:
            converted_df[col] = 1.0 / converted_df[col]
            converted_df = converted_df.rename(columns={col: inverse_ticker})

    return converted_df


def is_fx_for_ticker(ticker, for_ccy='USD'):
    """ If ticker is for FX spot and for_ccy is the foreign currency, answer True and
        the inverted ticker. Otherwise answer false and empty string """
    splits = str.split(ticker, " ")
    pair = splits[0]
    is_fx_for = False
    inverse_ticker = ""
    if len(pair) == 6 and pair[0:3].upper() == for_ccy:
        is_fx_for = True
        inverse_ticker = pair[3:6].upper() + pair[0:3].upper() + ticker[6:]

    return is_fx_for, inverse_ticker


if __name__ == "__main__":
    ROOT = r"C:\\temp\\sdevpy\\cointegration"
    FROM = '2015-07-23'
    TODAY = '2020-06-02'
    FROM_DATE = dt.date(2015, 7, 23)
    TO_DATE = dt.date(2020, 6, 2)
    ticker_list = ['GBPUSD Curncy', 'AUDUSD Curncy', 'NZDUSD Curncy', 'CADUSD Curncy', 'CNHUSD Curncy']

    # # Convert FX spots into USD and output converted data
    # data_file = os.path.join(ROOT, "test.tsv")
    # df_data = pd.read_csv(data_file, sep='\t')
    # df_data.set_index('Dates')
    # converted_df = convert_to_currency(df_data, "USD")
    # output_file = os.path.join(ROOT, "converted.tsv")
    # converted_df.to_csv(output_file, index=False, sep='\t')

    # New data
    data_file = os.path.join(ROOT, "fx_spots.tsv")
    df_raw = pd.read_csv(data_file, sep='\t')
    dates_str = df_raw['Dates']
    dates = [dt.datetime.strptime(x, "%Y-%m-%d").date() for x in dates_str]
    df_raw['Dates'] = dates
    df_fx_spots = df_raw[df_raw['Dates'] >= FROM_DATE]
    df_fx_spots = df_fx_spots[df_fx_spots['Dates'] <= TO_DATE]
    # print(df_data.to_string(max_rows=6, max_cols=6))

    # print(df_fx_spots.head())
    df_data = df_fx_spots[ticker_list]
    # print(df_data.head())
    test = johansen_test(df_data, 0, 1)
    # print(test)

    # # Old data
    # print("<><><><> Running OLD <><><><>")
    # data_file_xls = os.path.join(ROOT, "unit_test_data/bloomberg fx data sheet_for_unit_test.xlsx")
    # df_data_xls = myio.read_fx_daily_data(data_file_xls)
    # df_fx_spots_xls = df_data_xls.loc[FROM:TODAY]
    # df_fx_spots_xls = df_fx_spots_xls[ticker_list]

    # print(df_fx_spots_xls.head())
    # df_data_xls = df_fx_spots_xls[ticker_list]
    # print(df_data_xls.head())
    # res_test_xls = johansen_test(df_data_xls, 0, 1)
    # print(res_test_xls)

    # # Old function
    # print("<><><><><><><><> OLD <><><><><><><><>")
    # diagnostic_old = ct.johansen_test_diag(test, df_data, ticker_list, True, 0, 1)
    # print(diagnostic_old)

    # print("<><><><><><><><> NEW <><><><><><><><>")
    diagnostic = johansen_diagnostics(test, df_data, True, 0, 1)
    print(diagnostic)

