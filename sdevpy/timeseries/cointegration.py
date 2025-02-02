import os
import datetime as dt
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sdevpy.timeseries import meanreversion as mr

from sdevpy.cointegration import data_io as myio
from sdevpy.cointegration import coint_trading as ct

# [johansen_test_diag]
# Looks like we're going to compute the time series of the linear product and then fit
# a mean reversion to it. Then this no longer has any relation to the Johansen test.
# So wouldn't it be better to make this function more generic and focused on/named after
# what it does with the mean reversion?
def johansen_diagnostics(res_dict, data, asset_name_list, print = True):#, det_order = 0, k_ar_diff = 1):
    """ Compute the diagnostics of a basket. res_dict - from johansen_test_estimation """
    bool_trace_5pct = res_dict['trace (5%)']
    bool_trace_10pct = res_dict['trace (10%)'] 
    bool_eigen_5pct = res_dict['eigen (5%)'] 
    bool_eigen_10pct = res_dict['eigen (10%)'] 
    weights_XXXUSD = res_dict['weights']

    # get the data according to the asset name list 
    df_name_list = data[asset_name_list]

    # Compute the basket of currency with the estimated weights
    johansen_basket = pd.Series(np.dot(df_name_list, weights_XXXUSD), name='Basket') 
    johansen_basket.index = df_name_list.index

    # compute the mean reversion time series statistics 
    my_MeanRev_ts = mr.MeanRevTimeSeries(johansen_basket)

    mean_rev_level = my_MeanRev_ts.get_mean_rev_level() 
    mean_rev_rate_in_days = my_MeanRev_ts.get_mean_rev_rate_in_days() 
    time_in_days = my_MeanRev_ts.get_half_life_in_days()
    current_level = my_MeanRev_ts.get_current_level() 
    daily_hist_normal_vol = ut.compute_last_daily_hist_normal_vol(johansen_basket)
    current_zscore = my_MeanRev_ts.get_current_zscore()

    res_sharpe = mr.compute_sharpe_ratio(mean_rev_level, mean_rev_rate_in_days, time_in_days, 
                                         current_level, daily_hist_normal_vol, current_zscore)

    sharpe_ratio_half_life = res_sharpe['Sharpe Ratio']

    half_life_in_days = my_MeanRev_ts.get_half_life_in_days() 
    basket_std = my_MeanRev_ts.get_stdev()    
    
    position_df, USD_amount = ut.create_position_df(df_name_list, weights_XXXUSD)

    what_you_should_trade = list(position_df['market convention notional'].values)
    what_you_should_trade_USD_amount = USD_amount
    
    rsi_14 = RSIIndicator(johansen_basket, 14).rsi().iloc[-1]

    # flip the sign of the basket if SD is +ve, to indicate that we sell the basket
    if current_zscore > 0: 
        for i in range(len(what_you_should_trade)):
            what_you_should_trade[i] = -what_you_should_trade[i]

        what_you_should_trade_USD_amount = -1* what_you_should_trade_USD_amount 
        
    if print:
        const_pvalue = round(my_MeanRev_ts.get_const_pvalue(), 8)
        Basket_pvalue = round(my_MeanRev_ts.get_Basket_pvalue(), 8)
        print(' -------------------------------------------------------')
        print(position_df)
        print('USD_amount = ' + str(round(USD_amount, 4)))
        print(' -------------------------------------------------------')
        print('Pass trace test (5%) = ' + str(bool_trace_5pct))
        print('Pass eigen test (5%) = ' + str(bool_eigen_5pct))
        print('Pass trace test (10%) = ' + str(bool_trace_10pct))
        print('Pass eigen test (10%) = ' + str(bool_eigen_10pct))        
        res_jo = coint_johansen(df_name_list, det_order, k_ar_diff)
        print(trace_stats(res_jo)) 
        print(eigen_stats(res_jo))
        print(' -------------------------------------------------------')        
        print('half life in days = ' + str(round(half_life_in_days, 3)))
        print('const p-value = ' + str(const_pvalue))
        print('Basket p-value = ' + str(Basket_pvalue))
        print('current zscore = ' + str(round(current_zscore, 3)))
        print('SD Sharpe Ratio = ' + str(round(sharpe_ratio_half_life, 3)))
        print('RSI 14 = ' + str(round(rsi_14, 3)))
        print('current level = ' + str(round(current_level, 3)))
        print('1mio 1SD in USD = ' + str(round(settings.ONE_MILLION * basket_std, 0)))
        print('basket_std = ' + str(basket_std))
        print('mean_rev_level = ' + str(mean_rev_level))
        print('mean_rev_rate_in_days = ' + str(my_MeanRev_ts.get_mean_rev_rate_in_days() ))

    rounded_weights_XXXUSD = [round(w, 8) for w in weights_XXXUSD]

    # round PX_LAST and what_you_should_trade to 4 decimal for easy printing
    PX_LAST = list(position_df['PX_LAST'].values) 
    what_you_should_trade = [round(w, 4) for w in what_you_should_trade]
    PX_LAST = [round(w, 4) for w in PX_LAST]

    return {'half life in days': half_life_in_days, 'rounded weights': rounded_weights_XXXUSD, 
            'what you should trade': what_you_should_trade, 
            'what you should trade USD amount': what_you_should_trade_USD_amount,
            'current zscore': current_zscore,
            'trace (5%)': bool_trace_5pct, 'eigen (5%)': bool_eigen_5pct, 
            'trace (10%)': bool_trace_10pct, 'eigen (10%)': bool_eigen_10pct, 
            'Johansen Basket': johansen_basket,
            '1mio 1SD in USD': settings.ONE_MILLION * basket_std, 
            'PX_LAST' : PX_LAST, 'basket_std': basket_std, 'mean_rev_level': mean_rev_level,
            'half life Sharpe Ratio': sharpe_ratio_half_life, 'RSI 14': rsi_14}

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
    res_test = johansen_test(df_data, 0, 1)
    # print(res_test)

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


    res_diag = ct.johansen_test_diag(res_test, df_data, ticker_list, False, 0, 1)
    print(res_diag)

