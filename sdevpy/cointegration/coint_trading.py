import numpy as np
import pandas as pd
from sdevpy.cointegration import utils as ut
from sdevpy.cointegration import mean_reversion as my_mean_rev
from itertools import combinations
from tqdm import tqdm # for console progress bar
from sdevpy.cointegration import model_settings as settings
from ta.momentum import RSIIndicator
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_test_estimation(data, asset_name_list, det_order = 0, k_ar_diff = 1):
    """ Estimate the weights and Test statistics """
    # get the data according to the asset name list
    df_name_list = data[asset_name_list]

    # run johansen test
    res_jo = coint_johansen(df_name_list, det_order, k_ar_diff)

    # check the trace and eigenvalue test of Johansen
    bool_trace_5pct, bool_trace_10pct, bool_eigen_5pct, bool_eigen_10pct = check_johansen_test_stats_fast(res_jo)

    # get the normalized lst eigenvector which is the weights
    weights_xxxusd = norm_1st_eigvec(res_jo)

    res_dict = {'weights': weights_xxxusd,
                'trace (5%)': bool_trace_5pct,
                'eigen (5%)': bool_eigen_5pct,
                'trace (10%)': bool_trace_10pct,
                'eigen (10%)': bool_eigen_10pct
                }

    return res_dict


def johansen_test_diag(res_dict, data, asset_name_list, print_diagnostics = True, det_order = 0, k_ar_diff = 1):
    """ Compute the diagnostics of a basket. res_dict - from johansen_test_estimation """
    bool_trace_5pct = res_dict['trace (5%)']
    bool_trace_10pct = res_dict['trace (10%)']
    bool_eigen_5pct = res_dict['eigen (5%)']
    bool_eigen_10pct = res_dict['eigen (10%)']
    weights_xxxusd = res_dict['weights']

    # get the data according to the asset name list
    df_name_list = data[asset_name_list]

    # compute the basket of currency with the estimated weights
    johansen_basket = pd.Series(np.dot(df_name_list, weights_xxxusd), name='Basket')
    johansen_basket.index = df_name_list.index

    # compute the mean reversion time series statistics
    my_meanrev_ts = my_mean_rev.MeanRevTimeSeries(johansen_basket)

    mean_rev_level = my_meanrev_ts.get_mean_rev_level()
    mean_rev_rate_in_days = my_meanrev_ts.get_mean_rev_rate_in_days()
    time_in_days = my_meanrev_ts.get_half_life_in_days()
    current_level = my_meanrev_ts.get_current_level()
    daily_hist_normal_vol = ut.compute_last_daily_hist_normal_vol(johansen_basket)
    current_zscore = my_meanrev_ts.get_current_zscore()

    res_sharpe = my_mean_rev.compute_sharpe_ratio(mean_rev_level,
                                                  mean_rev_rate_in_days,
                                                  time_in_days,
                                                  current_level,
                                                  daily_hist_normal_vol,
                                                  current_zscore)

    sharpe_ratio_half_life = res_sharpe['Sharpe Ratio']

    half_life_in_days = my_meanrev_ts.get_half_life_in_days()
    basket_std = my_meanrev_ts.get_stdev()

    position_df, usd_amount = ut.create_position_df(df_name_list, weights_xxxusd)

    what_you_should_trade = list(position_df['market convention notional'].values)
    what_you_should_trade_usd_amount = usd_amount

    rsi_14 = RSIIndicator(johansen_basket, 14).rsi().iloc[-1]

    # flip the sign of the basket if SD is +ve, to indicate that we sell the basket
    if current_zscore > 0:
        for i in range(len(what_you_should_trade)):
            what_you_should_trade[i] = -what_you_should_trade[i]

        what_you_should_trade_usd_amount = -1* what_you_should_trade_usd_amount

    if print_diagnostics:
        const_pvalue = round(my_meanrev_ts.get_const_pvalue(), 8)
        basket_pvalue = round(my_meanrev_ts.get_Basket_pvalue(), 8)

        print(' -------------------------------------------------------')
        print(position_df)
        print('usd_amount = ' + str(round(usd_amount, 4)))
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
        print('Basket p-value = ' + str(basket_pvalue))
        print('current zscore = ' + str(round(current_zscore, 3)))
        print('SD Sharpe Ratio = ' + str(round(sharpe_ratio_half_life, 3)))
        print('RSI 14 = ' + str(round(rsi_14, 3)))
        print('current level = ' + str(round(current_level, 3)))
        print('1mio 1SD in usd = ' + str(round(settings.ONE_MILLION * basket_std, 0)))
        print('basket_std = ' + str(basket_std))
        print('mean_rev_level = ' + str(mean_rev_level))
        print('mean_rev_rate_in_days = ' + str(my_meanrev_ts.get_mean_rev_rate_in_days()))

    rounded_weights_xxxusd = [round(w, 8) for w in weights_xxxusd]

    # round PX_LAST and what_you_should_trade to 4 decimal for easy printing
    px_last = list(position_df['PX_LAST'].values)
    what_you_should_trade = [round(w, 4) for w in what_you_should_trade]
    px_last = [round(w, 4) for w in px_last]

    res_dict = {'half life in days': half_life_in_days,
                'rounded weights': rounded_weights_xxxusd,
                'what you should trade': what_you_should_trade,
                'what you should trade usd amount': what_you_should_trade_usd_amount,
                'current zscore': current_zscore,
                'trace (5%)': bool_trace_5pct,
                'eigen (5%)': bool_eigen_5pct,
                'trace (10%)': bool_trace_10pct,
                'eigen (10%)': bool_eigen_10pct,
                'Johansen Basket': johansen_basket,
                '1mio 1SD in usd': settings.ONE_MILLION * basket_std,
                'PX_LAST' : px_last,
                'basket_std': basket_std,
                'mean_rev_level': mean_rev_level,
                'half life Sharpe Ratio': sharpe_ratio_half_life,
                'RSI 14': rsi_14
                }

    return res_dict


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


def norm_1st_eigvec(res_jo):
    """ Retrieve the normalized first eigen vector from the result of Johansen test.
        The normalized vector are the weights of the cointegrated basket """
    # Size of the eigenvector
    n = len(res_jo.evec)
    first_eigvec = []
    for i in range(n):
        first_eigvec.append(res_jo.evec[i][0])

    # Normalized with respect to the first element
    res = np.array(first_eigvec)/res_jo.evec[0][0]
    return res


def check_johansen_test_stats_fast(res_jo):
    """ Extract the test stats for trace/eigen at 5% and 10% confident intervals """
    # trace_stats
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

   # eigen stats
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


def johansen_compute_all_baskets(start_list, today, all_name_list, df_fx_spot):
    """ Main routine in co-integration search.
        Given a list of start dates, today, list of currency pairs and the fx spot data,
        compute the Johansen test for all baskets
        start_list - e.g. ['2010-08-23', '2011-01-23']
        today - e.g. '2020-05-25'
        all_name_list - ['EURusd Curncy', 'GBPusd Curncy', 'AUDusd Curncy']
        df_fx_spot - DataFrame which contains all fx spot data """
    place_holder = 0.001
    empty_list = []

    # to create a no CNH list
    no_cnh_list = all_name_list.copy()
    no_cnh_list.remove('CNHusd Curncy')

    # The list to store all results
    baskets = []

    num_baskets = 0

    # tqdm is a progress bar functionality
    for start in tqdm(start_list):
        df_fx_spot_truncated = df_fx_spot.loc[start:today]

        # print('From ' + start + ' to ' + today )
        for number_ccy_pairs in range(3, 7):

            # Create all all N-tuple combinations
            sd = pd.to_datetime(start, format='%Y-%m-%d')

            if sd >= settings.CNH_BEGINNING_DATE:
                comb = combinations(all_name_list, number_ccy_pairs)
            else:
                # usdCNH only exists after 2010-08-23
                comb = combinations(no_cnh_list, number_ccy_pairs)

            # perform Johansen test for all N-tuple combinations
            for tup in list(comb):
                name_list = list(tup)

                # print_diagnostics = False, det_order = 0, k_ar_diff = 1
                res_estimation = johansen_test_estimation(df_fx_spot_truncated, name_list, 0, 1)

                # store the results
                baskets.append((
                                place_holder,   #'SD Current'
                                place_holder,   #'half life in days'
                                place_holder,   #'1mio 1SD in usd'
                                place_holder,   #'Stop Loss in SD' by compute_historical_min_max_SD_diagnostics
                                place_holder,   #'half life Sharpe Ratio'
                                place_holder,   #'RSI 14'
                                place_holder,   #'Top 3 SD Min' by compute_historical_min_max_SD_diagnostics
                                place_holder,   #'Top 3 SD Max' by compute_historical_min_max_SD_diagnostics
                                name_list,
                                empty_list,     #'what you should trade'
                                place_holder,   #'what you should trade usd amount'
                                start,
                                today,
                                res_estimation['weights'],
                                place_holder,  #'mean_rev_level'
                                place_holder,  #'basket_std'
                                res_estimation['trace (5%)'],
                                res_estimation['trace (10%)'],
                                res_estimation['eigen (5%)'],
                                res_estimation['eigen (10%)'],
                                place_holder,  #'Range in SD current' by compute_johansen_stability_diagnostics
                                place_holder,  #'+/- 1 month trace (5%)' by compute_johansen_stability_diagnostics
                                place_holder,  #'+/- 1 month trace (10%)' by compute_johansen_stability_diagnostics
                                place_holder,  #'+/- 1 month eigen (5%)' by compute_johansen_stability_diagnostics
                                place_holder,  #'+/- 1 month eigen (10%)' by compute_johansen_stability_diagnostics
                                ))

                num_baskets += 1
                #----------- END OF for tup in list(comb):

            #--------- END OF for number_ccy_pairs in range(3, 7):

    # store all the results in DataFrame and then output to Excel
    res_df = pd.DataFrame(baskets, columns =[
                                             'SD Current',
                                             'half life in days',
                                             '1mio 1SD in usd',
                                             'Stop Loss in SD',
                                             'half life Sharpe Ratio',
                                             'RSI 14',
                                             'Top 3 SD Min',
                                             'Top 3 SD Max',
                                             'currency pairs',
                                             'what you should trade',
                                             'what you should trade usd amount',
                                             'From',
                                             'Today',
                                             'unadj weights in XXXusd',
                                             'mean_rev_level',
                                             'basket_std',
                                             'trace (5%)',
                                             'trace (10%)',
                                             'eigen (5%)',
                                             'eigen (10%)',
                                             'Range in SD current',
                                             '+/- 1 month trace (5%)',
                                             '+/- 1 month trace (10%)',
                                             '+/- 1 month eigen (5%)',
                                             '+/- 1 month eigen (10%)'
                                             ])

    return res_df


def filter_cointegration_basket_using_trace_10(res_df):
    """ Select the cointegrated baskets, we use trace 10% as the minimum requirement
        res_df - output from the function, johansen_compute_all_baskets """
    trace_condition = res_df['trace (10%)']
    res_df_filtered = res_df[trace_condition]

    return res_df_filtered


def compute_johansen_test_diag_for_all_coint_baskets(res_df_filtered, df_fx_spot):
    """ Compute the diagnostics only for cointegrated basket """
    for ind in tqdm(res_df_filtered.index):
        from_ = res_df_filtered['From'][ind]
        today_ = res_df_filtered['Today'][ind]

        df_fx_spot_truncated = df_fx_spot.loc[from_:today_]

        name_list = res_df_filtered['currency pairs'][ind]

        res_estimation = {'weights': res_df_filtered['unadj weights in XXXusd'][ind],
                          'trace (5%)': res_df_filtered['trace (5%)'][ind],
                          'eigen (5%)': res_df_filtered['eigen (5%)'][ind],
                          'trace (10%)': res_df_filtered['trace (10%)'][ind],
                          'eigen (10%)': res_df_filtered['eigen (10%)'][ind]
                         }

        res_diag = johansen_test_diag(res_estimation, df_fx_spot_truncated, name_list, False, 0, 1)

        res_df_filtered['unadj weights in XXXusd'][ind] = res_diag['rounded weights']
        res_df_filtered['SD Current'][ind] = round(res_diag['current zscore'], 3)
        res_df_filtered['half life in days'][ind] = round(res_diag['half life in days'], 3)
        res_df_filtered['1mio 1SD in usd'][ind] = round(res_diag['1mio 1SD in usd'], 0)
        res_df_filtered['half life Sharpe Ratio'][ind] = res_diag['half life Sharpe Ratio']
        res_df_filtered['RSI 14'][ind] = res_diag['RSI 14']
        res_df_filtered['what you should trade'][ind] = res_diag['what you should trade']
        res_df_filtered['what you should trade usd amount'][ind] = res_diag['what you should trade usd amount']
        res_df_filtered['mean_rev_level'][ind] = res_diag['mean_rev_level']
        res_df_filtered['basket_std'][ind] = res_diag['basket_std']

    return res_df_filtered


def filter_cointegration_basket_using_sd_threshold(res_df, sd_threshold):
    #----------------------------------------------------------------------------------------
    # Choose the basket such that the current absolute of stdev is above SD_threshold and it is cointegrated
    # res_df - output from the function, johansen_compute_all_baskets
    # SD_threshold - we filter out basket with SD less than abs(SD_threshold). If SD_threshold = 2,
    # we only show baskets with abs(SD) > 2
    # select using SD criteria
    upper_distance_condition = res_df['SD Current'] > sd_threshold
    lower_distance_condition = res_df['SD Current'] < -sd_threshold
    res_df_filtered = res_df[upper_distance_condition | lower_distance_condition]

    # sort with SD
    res_df_filtered = res_df_filtered.sort_values('SD Current', ascending=False)

    return res_df_filtered


def filter_cointegration_basket_using_1m_1sd_in_usd(res_df):
    # Choose the basket such that '1mio 1SD in usd' < 100000
    one_mio_1sd_condition = res_df['1mio 1SD in usd'] < 100000
    res_df_filtered = res_df[one_mio_1sd_condition]
    return res_df_filtered


def filter_cointegration_basket_using_sharpe_ratio(res_df, sharpe_threshold):
    # Choose the basket using Sharpe Ratio
    sr_condition = res_df['half life Sharpe Ratio'] > sharpe_threshold
    res_df_filtered = res_df[sr_condition]
    return res_df_filtered


def filter_cointegration_basket_using_half_life_in_days(res_df, half_life_threshold):
    # Choose the basket using half life
    hl_condition = res_df['half life in days'] < half_life_threshold
    res_df_filtered = res_df[hl_condition]
    return res_df_filtered


def compute_historical_min_max_sd_diagnostics(res_df_filtered, df_fx_spot):
    #--- Compute historical SD for the filtered basket
    #
    # df_fx_spot - fx spot data to be used to compute diagnostics
    #
    # This function modifies the input dataframe - res_df_filtered of the following columns
    # (1) 'Stop Loss in SD'
    # (2) 'Top 3 SD Min'
    # (3) 'Top 3 SD Max'
    for ind in tqdm(res_df_filtered.index):

        from_ = res_df_filtered['From'][ind]
        today_ = res_df_filtered['Today'][ind]

        # get the full list fx spot time series
        df_from_today = df_fx_spot.loc[from_:today_]

        name_list = res_df_filtered['currency pairs'][ind]
        weights_xxxusd = res_df_filtered['unadj weights in XXXusd'][ind]

        z_score_ts = my_mean_rev.compute_zscore_for_a_fixed_basket_for_multi_period(df_from_today,
                                                                                    name_list,
                                                                                    weights_xxxusd)

        # sort the zscore in descending order
        sorted_zscore = z_score_ts.sort_values(ascending=False)

        sd_current = res_df_filtered['SD Current'][ind]
        if sd_current > 0:
            max_sd = sorted_zscore.iloc[0]
            stop_loss_in_sd = (max_sd - sd_current)
        elif sd_current < 0:
            min_sd = sorted_zscore.iloc[-1]
            stop_loss_in_sd = (sd_current - min_sd)

        res_df_filtered['Stop Loss in SD'][ind] = round(stop_loss_in_sd, 3)

        # ------------------------------------------
        min_list = []
        for i in range(1, 4):
            sd = round( sorted_zscore.iloc[-i], 2)
            date = sorted_zscore.index[-i].strftime('%Y-%m-%d')
            min_list.append((sd, date))

        res_df_filtered['Top 3 SD Min'][ind] = min_list

        max_list = []
        for i in range(0, 3):
            sd = round( sorted_zscore.iloc[i], 2)
            date = sorted_zscore.index[i].strftime('%Y-%m-%d')
            max_list.append((sd, date))

        res_df_filtered['Top 3 SD Max'][ind] = max_list

    return res_df_filtered


def remove_data_within_x_days(df, today):
    # helper function to remove the most recent x days of in the time series
    ts_today = pd.Timestamp(today)
    bd = pd.tseries.offsets.BusinessDay(n = settings.HOLDING_PERIOD_IN_DAYS-1)
    x_minus_1_days_ago = ts_today - bd

    filter_condition = df.index < x_minus_1_days_ago
    return df[filter_condition]


def extract_data_conditions_on_buy_sell_signal(time_series, df_buy_sell_signal, today):
    # -- Given an array data and buy sell signal, extract the following arrays
    # (1) dates_to_buy - dates that we long the basket
    # (2) retns_to_buy - time series corresponds to dates_to_buy
    # (3) dates_to_sell - dates that we short the basket
    # (4) retns_to_sell - time series corresponds to dates_to_sell
    df_for_plot = pd.concat([time_series, df_buy_sell_signal], axis=1)
    df_for_plot = df_for_plot.fillna(0)

    # remove the last 4 days data as we are computing 5 days returns.
    # They will be 0 (because the data is not there yet) and will change the average and stdev
    df_for_plot = remove_data_within_x_days(df_for_plot, today)

    column_name = time_series.name

    dates_to_buy = df_for_plot.loc[df_for_plot['Buy Signal'] ==1].index
    retns_to_buy = df_for_plot.loc[df_for_plot['Buy Signal'] ==1, column_name].values

    # need to remove the last 5 days to have an accurate estimate

    dates_to_sell = df_for_plot.loc[df_for_plot['Sell Signal'] ==1].index
    retns_to_sell = df_for_plot.loc[df_for_plot['Sell Signal'] ==1, column_name].values

    return dates_to_buy, retns_to_buy, dates_to_sell, retns_to_sell


def date_formatter(date, datafreq):
    if datafreq == settings.DataFreq.DAILY:
        date_formatted = pd.to_datetime(date, format='%Y-%m-%d')
    elif datafreq == settings.DataFreq.INTRADAY:
        date_formatted = pd.to_datetime(date, format='%Y-%m-%d, %H-%M-%S')

    return date_formatted


def range_start_range_end_for_stability_test(from_, today_, name_list, jvd, datafreq):
    if datafreq == settings.DataFreq.DAILY:
        if jvd == settings.JohansenVaringDate.END:
            range_start = date_formatter(today_, datafreq) - pd.DateOffset(months=1)
            range_end = date_formatter(today_, datafreq)
        elif jvd == settings.JohansenVaringDate.START:
            range_start = date_formatter(from_, datafreq) - pd.DateOffset(months=1)
            range_end = date_formatter(from_, datafreq) + pd.DateOffset(months=1)
    elif datafreq == settings.DataFreq.INTRADAY:
        if jvd == settings.JohansenVaringDate.END:
            range_start = date_formatter(today_, datafreq) - pd.DateOffset(days=10)
            range_end = date_formatter(today_, datafreq)
        elif jvd == settings.JohansenVaringDate.START:
            range_start = date_formatter(from_, datafreq)
            range_end = date_formatter(from_, datafreq) + pd.DateOffset(days=10)
    return range_start, range_end


def compute_johansen_params_stability(df_fx_spot, from_, today_, name_list, jvd, datafreq):
    # ----- Compute the z_score_stability by changing the data start date in 2 months window
    #       We check the stability by rerun Johansen test using different data.
    output = []

    range_start, range_end = range_start_range_end_for_stability_test(from_, today_, name_list, jvd, datafreq)

    for t in df_fx_spot.index:
        if range_start < t and t <= range_end:

            if jvd == settings.JohansenVaringDate.END:
                df_fx_truncated = df_fx_spot[from_:t]
            elif jvd == settings.JohansenVaringDate.START:
                df_fx_truncated = df_fx_spot[t:today_]

            res_estimation = johansen_test_estimation(df_fx_truncated, name_list, 0, 1)
            res_diag = johansen_test_diag(res_estimation, df_fx_truncated, name_list, False, 0, 1)

            # store the results
            if jvd == settings.JohansenVaringDate.END:
                from_date_ouptput = date_formatter(from_, datafreq)
                end_date_output = date_formatter(t, datafreq)
            elif jvd == settings.JohansenVaringDate.START:
                from_date_ouptput = date_formatter(t, datafreq)
                end_date_output = date_formatter(today_, datafreq)

            output.append((from_date_ouptput,
                           end_date_output,
                           res_estimation['trace (5%)'],
                           res_estimation['eigen (5%)'] ,
                           res_estimation['trace (10%)'],
                           res_estimation['eigen (10%)'],
                           round(res_diag['half life in days'], 2),
                           round(res_diag['current zscore'], 2),
                           name_list,
                           res_diag['rounded weights'],
                           res_diag['what you should trade'],
                           res_diag['PX_LAST'],
                           res_diag['basket_std'],
                           res_diag['mean_rev_level']
                           ))
            # ----------END OF if RANGE_START < t and t <= RANGE_END:
        # --------- END OF for t in df_fx_spot.index:

    # store all the results in DataFrame and then output to Excel
    res_df = pd.DataFrame(output, columns =['From',
                                            'Current Date',
                                            'trace (5%)',
                                            'eigen (5%)',
                                            'trace (10%)',
                                            'eigen (10%)',
                                            'half life in days',
                                            'current zscore',
                                            'currency pairs',
                                            'unadj weights in xxxusd',
                                            'what you should trade',
                                            'PX_LAST',
                                            'basket_std',
                                            'mean_rev_level'
                                            ])

    return res_df


def compute_johansen_stability_diagnostics(res_df_filtered, df_fx_spot, datafreq):
    #--- Compute stability test by changing the start date of the dataset.
    # diagnostics l - current SD (by compute the range of SD, max(SD) - min(SD)
    # diagnostics 2 - trace and eigne tests (return if all the them are true)
    # res_df_filtered - output from the function, filter_cointegration_basket
    # (1) 'Range in SD current'
    # (2) '+/1 month trace (5%)'
    # (3) '+/1 month trace (10%)'
    # (4) '+/1 eigen trace (5%)'
    # (5) '+/1 eigen trace (10%)'
    for ind in tqdm(res_df_filtered.index):
        from_ = res_df_filtered['From'][ind]
        today_ = res_df_filtered['Today'][ind]
        name_list = res_df_filtered['currency pairs'][ind]
        res_start_df = compute_johansen_params_stability(df_fx_spot, from_, today_, name_list,
                                                         settings.JohansenVaringDate.START, datafreq)

        # diagnostics 1
        current_zscores = res_start_df['current zscore']
        range_in_zscore = np.max(current_zscores) - np.min(current_zscores)
        res_df_filtered['Range in SD current'][ind] = round(range_in_zscore, 3)

        # diagnostics 2
        trace_5pcts = res_start_df['trace (5%)']
        trace_10pcts = res_start_df['trace (10%)']
        eigen_5pcts = res_start_df['eigen (5%)']
        eigen_10pcts = res_start_df['eigen (10%)']

        # make sure the series is not empty as as all() function on an empty list is TRUE. Assume if one is not empty,
        # the rest are not empty.
        if len(trace_10pcts) == 0:
            raise Exception('the list, trace_10pcts, is empty, something is wrong.')

        # we only filter out if the number of True is less than 90%
        pct = 0.9

        res_df_filtered['+/- 1 month trace (5%)'][ind] = is_true_by_percentage(trace_5pcts.tolist(), pct)
        res_df_filtered['+/- 1 month trace (10%)'][ind] = is_true_by_percentage(trace_10pcts.tolist(), pct)
        res_df_filtered['+/- 1 month eigen (5%)'][ind] = is_true_by_percentage(eigen_5pcts.tolist(), pct)
        res_df_filtered['+/- 1 month eigen (10%)'][ind] = is_true_by_percentage(eigen_10pcts.tolist(), pct)
        # ---------------------

    res_df_filtered = filter_cointegration_basket_using_trace_10_stability(res_df_filtered)
    res_df_filtered = filter_cointegration_basket_using_range_in_sd_current(res_df_filtered)
    return res_df_filtered


def is_true_by_percentage(list_of_booleans, percentage):
    # count the number of True, divided by the total number of elements.
    # if higher than percentage then we return True
    num_true = list_of_booleans.count(True)
    total_num = len(list_of_booleans)

    if num_true/total_num > percentage:
        return True
    else:
        return False


def filter_cointegration_basket_using_trace_10_stability(res_df):
    # Choose the basket such that the cointegration condition, 'trace (10%)' = True are satisfied for +/- 1 month
    trace_condition = res_df['+/- 1 month trace (10%)']
    res_df_filtered = res_df[trace_condition]
    return res_df_filtered


def filter_cointegration_basket_using_range_in_sd_current(res_df):
    # Choose the basket such that 'Range in SD current' < 0.5. Bigger the range, more the instability.
    range_in_sd_current_condition = res_df['Range in SD current'] < 0.5
    res_df_filtered = res_df[range_in_sd_current_condition]
    return res_df_filtered
