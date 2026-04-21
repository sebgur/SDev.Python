import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from ta.momentum import RSIIndicator
from sdevpy.timeseries import meanreversion as mr
from sdevpy.timeseries import timeseriestools as tst
from sdevpy.utilities import clipboard as clipboard


def johansen_diagnostics(test: dict, df_data: pd.DataFrame, verbose: bool=True, det_order: int=0,
                         k_ar_diff: int=1) -> dict:
    """ Compute the diagnostics of a basket fom Johansen's test results.
        Looks like we're going to compute the time series of the linear product and then fit
        a mean reversion to it. Then this no longer has any relation to the Johansen test.
        So wouldn't it be better to make this function more generic and focused on/named after
        what it does with the mean reversion?
     """
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

    position_df, usd_amount = tst.create_position(df_data, weights)

    what_you_should_trade = list(position_df['market convention notional'].values)
    what_you_should_trade_usd_amount = usd_amount

    rsi_14 = RSIIndicator(johansen_basket, 14).rsi().iloc[-1]

    # Flip the sign of the basket if SD is +ve, to indicate that we sell the basket
    if current_zscore > 0:
        for i in range(len(what_you_should_trade)):
            what_you_should_trade[i] = -what_you_should_trade[i]

        what_you_should_trade_usd_amount = -1 * what_you_should_trade_usd_amount

    if verbose:
        const_pvalue = round(mr_ts.get_const_pvalue(), 8)
        series_pvalue = round(mr_ts.get_series_pvalue(), 8)
        print(' -------------------------------------------------------')
        print(position_df)
        print('USD_amount = ' + str(round(usd_amount, 4)))
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
    px_last = list(position_df['PX_LAST'].values)
    what_you_should_trade = [round(w, 4) for w in what_you_should_trade]
    px_last = [round(w, 4) for w in px_last]

    return {'Half life': half_life, 'Rounded weights': rounded_weights,
            'What you should trade': what_you_should_trade,
            'What you should trade USD amount': what_you_should_trade_usd_amount,
            'Current zscore': current_zscore,
            'Trace (5%)': bool_trace_5pct, 'Eigen (5%)': bool_eigen_5pct,
            'Trace (10%)': bool_trace_10pct, 'Eigen (10%)': bool_eigen_10pct,
            'Johansen Series': johansen_basket, '1mio 1SD in USD': 1e6 * series_stdev,
            'PX_LAST' : px_last, 'Series stdev': series_stdev, 'MR level': mr_level,
            'Half life Sharpe Ratio': sharpe_ratio_half_life, 'RSI 14': rsi_14}


def johansen_test(df_data: pd.DataFrame, det_order: int=0, k_ar_diff: int=1) -> dict:
    """ Estimate the weights and test statistics """
    # Run Johansen test
    res_jo = coint_johansen(df_data, det_order, k_ar_diff)

    # Check the trace and eigenvalue test of Johansen
    trace_5pct, trace_10pct, eigen_5pct, eigen_10pct = check_johansen_stats_fast(res_jo)

    # Get the normalized lst eigenvector which is the weights
    weights = norm_1st_eigvec(res_jo)

    return {'weights': weights, 'trace (5%)': trace_5pct, 'eigen (5%)': eigen_5pct,
            'trace (10%)': trace_10pct, 'eigen (10%)': eigen_10pct}


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
    n = len(res_jo.lr1)

    data = np.zeros((n, 4))

    # create the row labels
    index_list = ['r=0']
    for i in range(1, n):
        index_list.append('r<=' + str(i))

    # populate the data in the desired format
    for i in range(n):
        data[i][0] = res_jo.lr1[i]
        for j in range(3):
            data[i][j+1] = res_jo.cvt[i][j]

    res_df = pd.DataFrame(data, columns = ['trace', '10%', '5%', '1%'], index = index_list)
    return res_df


def eigen_stats(res_jo):
    """ Retrieve the eigen test statistics from the result of Johansen test in a pretty format """
    # Find out the size of the vector. This is the number of assets.
    n = len(res_jo.lr2)

    data = np.zeros((n, 4))

    # create the row labels
    index_list = ['r=0']
    for i in range(1, n):
        index_list.append('r<=' + str(i))

    # populate the data in the desired format
    for i in range(n):
        data[i][0] = res_jo.lr2[i]
        for j in range(3):
            data[i][j+1] = res_jo.cvm[i][j]

    res_df = pd.DataFrame(data, columns = ['eigen', '10%', '5%', '1%'], index = index_list)
    return res_df


def norm_1st_eigvec(res_jo) -> npt.ArrayLike:
    """ Retrieve the normalized first eigen vector from the result of Johansen test.
        The normalized vector are the weights of the cointegrated basket """
    # Size of the eigenvector
    n = len(res_jo.evec)
    first_eigvec = []
    for i in range(n):
        first_eigvec.append(res_jo.evec[i][0])

    # Normalized with respect to the first element
    return np.array(first_eigvec) / res_jo.evec[0][0]


if __name__ == "__main__":
    ROOT = r"C:\\temp\\sdevpy\\cointegration"
    FROM = '2015-07-23'
    TODAY = '2020-06-02'
    tickers = ['GBPUSD Curncy', 'AUDUSD Curncy', 'NZDUSD Curncy', 'CADUSD Curncy', 'CNHUSD Curncy']

    data_file = os.path.join(ROOT, "fx_spots.tsv")
    df_raw = pd.read_csv(data_file, sep='\t')
    df_raw.set_index('Dates', inplace=True)
    df_fx_spots = df_raw.loc[FROM:TODAY]
    df_data = df_fx_spots[tickers]
    test = johansen_test(df_data, 0, 1)
    diagnostic = johansen_diagnostics(test, df_data, True, 0, 1)
    print(diagnostic)
