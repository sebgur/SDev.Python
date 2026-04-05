import numpy as np
import pandas as pd
# import statsmodels.api as sm
# from sdevpy.cointegration import coint_trading as ct
from sdevpy.cointegration import utils as ut
# from sdevpy.cointegration import data_io as myio
# from sdevpy.cointegration import mean_reversion as my_mean_rev
import math
from tqdm import tqdm # for console progress bar

# input string
# "['EURUSD Curncy', 'GBPUSD Curncy', 'JPYUSD Curncy', 'SGDUSD Curncy', 'CNHUSD Curncy']"
# we need to remove all "[", "]" and "'", then split by "," and then create a list
def name_list_string_to_name_list(name_list_str):
    res_str = name_list_str.replace(" '", "")
    res_str = res_str.replace("'", "")
    res_str = res_str.replace('[', '')
    res_str = res_str.replace(']', '')
    name_list = list(res_str.split(','))
    return name_list


def weigths_list_string_to_float_list(weigths_list_str):
    res_str = weigths_list_str.replace('[', '')
    res_str = res_str.replace(']', '')
    float_list = [float(idx) for idx in res_str.split(',')]
    return float_list


# Given a fixed basket, from date, trade date, now date
# compute the sharpe ratio, z_score, 5D realized return and 10D realized return from trade date
def back_test_one_trade(from_, trade_date, now, name_list, weights_xxxusd, zscore_trade_date, df_fx_spot):
    df_fx_name_list_xxxusd = df_fx_spot[name_list]

    # compute the basket up to NOW, because we need to compute the realized 5D and 10D returns
    basket_up_to_now = ut.compute_basket(df_fx_name_list_xxxusd.loc[from_:now], weights_xxxusd)

    # compute the stdev as if we were on the TRADE DATE, this is used for computing the return in terms of ZScore
    stdev_up_to_trade_date = np.std(basket_up_to_now.loc[:trade_date])

    basket_from_trade_date_to_now = basket_up_to_now.loc[trade_date:now]

    basket_2d_rtns_from_trade_date = ut.compute_x_day_historical_returns_in_SD(basket_from_trade_date_to_now, 2,
     stdev_up_to_trade_date).iloc[0]
    basket_5d_rtns_from_trade_date = ut.compute_x_day_historical_returns_in_SD(basket_from_trade_date_to_now, 5,
     stdev_up_to_trade_date).iloc[0]
    basket_10d_rtns_from_trade_date = ut.compute_x_day_historical_returns_in_SD(basket_from_trade_date_to_now, 10,
     stdev_up_to_trade_date).iloc[0]

    min_max_res_2d = ut.min_max_return_x_days_in_SD(basket_from_trade_date_to_now, 2, stdev_up_to_trade_date).iloc[0]
    min_2d_rtns_in_sd = min_max_res_2d['min']
    max_2d_rtns_in_sd = min_max_res_2d['max']

    min_max_res_5d = ut.min_max_return_x_days_in_SD(basket_from_trade_date_to_now, 5, stdev_up_to_trade_date).iloc[0]
    min_5d_rtns_in_sd = min_max_res_5d['min']
    max_5d_rtns_in_sd = min_max_res_5d['max']

    min_max_res_10d = ut.min_max_return_x_days_in_SD(basket_from_trade_date_to_now, 10, stdev_up_to_trade_date).iloc[0]
    min_10d_rtns_in_sd = min_max_res_10d['min']
    max_10d_rtns_in_sd = min_max_res_10d['max']

    if math.isnan(basket_10d_rtns_from_trade_date):
        raise Exception('Trade DATE is less than 10 days ago.')

    if zscore_trade_date > 0:
        # flip the return sign if zscore is above 0, because we sell. The basket goes down and we earn
        basket_2d_rtns_from_trade_date = -basket_2d_rtns_from_trade_date
        basket_5d_rtns_from_trade_date = -basket_5d_rtns_from_trade_date
        basket_10d_rtns_from_trade_date = -basket_10d_rtns_from_trade_date

        # so the max draw down is the negative of the max returns
        max_2d_draw_down_in_sd = -max_2d_rtns_in_sd
        max_5d_draw_down_in_sd = -max_5d_rtns_in_sd
        max_10d_draw_down_in_sd = -max_10d_rtns_in_sd
    else:
        # we buy so that max draw down is teh min returns
        max_2d_draw_down_in_sd = min_2d_rtns_in_sd
        max_5d_draw_down_in_sd = min_5d_rtns_in_sd
        max_10d_draw_down_in_sd = min_10d_rtns_in_sd

    #if the max draw down is a positive number, we floor it to 0 to show there is no loss
    max_2d_draw_down_in_sd = np.minimum(max_2d_draw_down_in_sd, 0.0)
    max_5d_draw_down_in_sd = np.minimum(max_5d_draw_down_in_sd, 0.0)
    max_10d_draw_down_in_sd = np.minimum(max_10d_draw_down_in_sd, 0.0)

    res_dict = {'2D Realized Rtns from Trade Date in SD': basket_2d_rtns_from_trade_date,
                '5D Realized Rtns from Trade Date in SD': basket_5d_rtns_from_trade_date,
                '10D Realized Rtns from Trade Date in SD': basket_10d_rtns_from_trade_date,
                '2D max draw down in SD': max_2d_draw_down_in_sd,
                '5D max draw down in SD': max_5d_draw_down_in_sd,
                '10D max draw down in SD': max_10d_draw_down_in_sd,
                'basket stdev on Trade Date': stdev_up_to_trade_date
                }

    return res_dict

# res_df_filtered - output from coint_trading.filter_cointegration_basket
def back_test_many_trades(res_df_filtered, now, df_fx_spot):
    back_test_res = []

    num_rows = len(res_df_filtered)

    for idx in tqdm(range(num_rows)):
        from_ = res_df_filtered['From'].iloc[idx]
        trade_date = res_df_filtered['Today'].iloc[idx]
        trade_date = pd.to_datetime(trade_date, format='%Y-%m-%d')

        name_str = res_df_filtered['currency pairs'].iloc[idx]
        name_list = name_list_string_to_name_list(name_str)

        weights_xxxusd_str = res_df_filtered['unadj weights in xxxusd'].iloc[idx]
        weights_xxxusd = weigths_list_string_to_float_list(weights_xxxusd_str)

        sharpe_5d = res_df_filtered['5D Sharpe Ratio'].iloc[idx]
        sd_on_trade_date = res_df_filtered['SD Current'].iloc[idx]

        stop_loss_in_sd = res_df_filtered['Stop Loss in SD'].iloc[idx]

        half_life_in_days = res_df_filtered['half life in days'].iloc[idx]

        range_in_sd_current = res_df_filtered['Range in SD current'].iloc[idx]

        abs_sd_on_trade_date = np.abs(sd_on_trade_date)

        one_month_trace_5pct = res_df_filtered['+/- 1 month trace (5%)'].iloc[idx]
        one_month_trace_10pct = res_df_filtered['+/- 1 month trace (10%)'].iloc[idx]
        one_month_eigen_5pct = res_df_filtered['+/- 1 month eigen (5%)'].iloc[idx]
        one_month_eigen_10pct = res_df_filtered['+/- 1 month eigen (10%)'].iloc[idx]

        res_dict = back_test_one_trade(from_, trade_date, now, name_list, weights_xxxusd,
                                       sd_on_trade_date, df_fx_spot)

        basket_stdev = res_dict['basket stdev on Trade Date']

        mean_rev_level = res_df_filtered['mean_rev_level'].iloc[idx]

        back_test_res.append((from_, trade_date, now, name_list, weights_xxxusd, sharpe_5d,
                              sd_on_trade_date, abs_sd_on_trade_date, stop_loss_in_sd,
                              res_dict['2D Realized Rtns from Trade Date in SD'],
                              res_dict['5D Realized Rtns from Trade Date in SD'],
                              res_dict['10D Realized Rtns from Trade Date in SD'],
                              res_dict['2D max draw down in SD'],
                              res_dict['5D max draw down in SD'],
                              res_dict['10D max draw down in SD'],
                              basket_stdev, mean_rev_level, half_life_in_days, range_in_sd_current,
                              one_month_trace_5pct, one_month_trace_10pct,
                              one_month_eigen_5pct, one_month_eigen_10pct))

        #--- end of for ind in tqdm(res_df_filtered.index):

    res_df = pd.DataFrame(back_test_res, columns =['from_', 'Trade Date', 'Now', 'currency pairs',
                                                   'unadj weights in xxxusd', 'Sharpe Ratio on Trade Date',
                                                   'Z Score on Trade Date', 'Abs Z Score on Trade Date',
                                                   'Distance to max/min SD', '2D Realized Rtns in SD',
                                                   '5D Realized Rtns in SD', '10D Realized Rtns in SD',
                                                   '2D max DD in SD', '5D max DD in SD', '10D max DD in SD',
                                                   'basket stdev on Trade Date', 'mean_rev_level on Trade Date',
                                                   'half life in days', 'Range in SD current',
                                                   '+/- 1 month trace (5%)', '+/- 1 month trace (10%)',
                                                   '+/- 1 month eigen (5%)', '+/- 1 month eigen (10%)'])

    return res_df

# Take the results of back_test_many_trades and then compute the back test diagnostics
def one_back_test_summary_table(back_test_res_df, start_date, end_date, sharpe_threshold, zscore_threshold):
    upper_sd_condition = back_test_res_df['Z Score on Trade Date'] > zscore_threshold
    lower_sd_condition = back_test_res_df['Z Score on Trade Date'] < -zscore_threshold
    sharpe_condition = back_test_res_df['Sharpe Ratio on Trade Date'] > sharpe_threshold

    start_date_condition = back_test_res_df['Trade Date'] > pd.Timestamp(start_date)
    end_date_condition = back_test_res_df['Trade Date'] < pd.Timestamp(end_date)

    # -----------apply conditions
    my_trade_df = back_test_res_df[sharpe_condition]
    my_trade_df = my_trade_df[upper_sd_condition | lower_sd_condition]
    my_trade_df = my_trade_df[start_date_condition]
    my_trade_df = my_trade_df[end_date_condition]

    total = len(my_trade_df)

    output_table = []

    column_list = ['2D Realized Rtns in SD', '5D Realized Rtns in SD', '10D Realized Rtns in SD']

    for column in column_list:
        positive_return_condition = my_trade_df[column] > 0.0
        num_pos_rtn = len(my_trade_df[positive_return_condition])
        num_neg_rtn = len(my_trade_df[~positive_return_condition])

        pos_rtn_pct = round(num_pos_rtn/total, 3)
        neg_rtn_pct = round(num_neg_rtn/total, 3)

        rtns_median = np.median(my_trade_df[column])
        rtns_mean = np.mean(my_trade_df[column])
        rtns_std = np.std(my_trade_df[column])

        first_3_char = column[:3]

        output_table.append((start_date,
                             end_date,
                             sharpe_threshold,
                             zscore_threshold,
                             first_3_char,
                             pos_rtn_pct,
                             neg_rtn_pct,
                             total,
                             rtns_median,
                             rtns_mean,
                             rtns_std))

    res_df = pd.DataFrame(output_table, columns =['Period Start',
                                                  'Period End',
                                                  'Sharpe threshold',
                                                  'SD threshold',
                                                  'Rtns type',
                                                  'Pos %',
                                                  'Neg %',
                                                  'Total',
                                                  'Rtns median',
                                                  'Rtns mean',
                                                  'Rtns stdev'
                                                  ])

    return res_df, my_trade_df
