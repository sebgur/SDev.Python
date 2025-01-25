import numpy as np
import pandas as pd
import utils as ut
import mean_reversion as my_mean_rev
import coint_trading as ct
import back_testing as btest
import matplotlib.pyplot as plt
import seaborn as sns
import model_settings as settings
       
# ---------------------------------------------------------------------------
# x_lim_tuple contains (start_date, end_date)
def normal_plot(time_series_basket, x_lim_tuple, vertline_date, q_pct, style_str='g-o'):
     
    ax = time_series_basket.plot(kind='line', grid=True, figsize=(15, 5), style=style_str)
    
    # set the x limits for plotting
    start_date = x_lim_tuple[0]
    end_date = x_lim_tuple[1]    
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    label = time_series_basket.columns[0]
    ax.set_ylabel(label)
    
    quantile = time_series_basket.quantile([q_pct/100.0, 1.0-q_pct/100.0])   
    
    # max
    ts_max = time_series_basket.max()
    plt.axhline(ts_max[label], color='b')
    
    # 1-q_pct quantile
    plt.axhline(quantile.iloc[1][label], color='c')
    
    # mean
    ts_mean = time_series_basket.mean()
    plt.axhline(ts_mean[label], color='g')
    
    # q_pct quantile
    plt.axhline(quantile.iloc[0][label], color='r')
    
    # min
    ts_min = time_series_basket.min()
    plt.axhline(ts_min[label], color='k')    

    plt.title(str(time_series_basket.iloc[-1]))
    
    plt.legend([label,                 
                'max:' + str(round(ts_max[label], 4)),   
                str(100-q_pct) + ' Q:' + str(round(quantile.iloc[1][label], 4)),  
                'mean:' + str(round(ts_mean[label], 4)),                 
                str(q_pct) + ' Q:' + str(round(quantile.iloc[0][label], 4)),  
                'min:' + str(round(ts_min[label], 4))
                ]);
                
    plt.axvline(vertline_date, color='magenta', linestyle='--')

    plt.show()
             
# ---------------------------------------------------------------------------
# x_lim_tuple contains (start_date, end_date)
def plot_hist_returns(basket_x_days_hist_rtns, df_buy_sell_signal, x_lim_tuple, z_score_threshold, vertline_date):
      
    hist_rtns_times_1mio = basket_x_days_hist_rtns * settings.ONE_MILLION
    
    ax = hist_rtns_times_1mio.plot(kind='line', grid=True, figsize=(15,5))
    
    plt.axvline(vertline_date, color='magenta', linestyle='--')
    
    # set the x limits for plotting
    start_date = x_lim_tuple[0]
    end_date = x_lim_tuple[1]    
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))         
    
    #-------------------------------------------------------------------------   
    TODAY = basket_x_days_hist_rtns.index[-1].strftime('%Y-%m-%d')
    
    dates_to_buy, retns_to_buy, dates_to_sell, retns_to_sell = ct.extract_data_conditions_on_buy_sell_signal(hist_rtns_times_1mio, 
                                                                                                             df_buy_sell_signal, 
                                                                                                             TODAY)
    if len(dates_to_buy) > 1:
        # plt.scatter dos not like if the array only has one element. That's why doing this check
        plt.scatter(dates_to_buy, retns_to_buy, label='skitscat', color='green', s=50, marker='^')
        
    if len(dates_to_sell) > 1:
        # plt.scatter dos not like if the array only has one element. That's why doing this check
        plt.scatter(dates_to_sell, retns_to_sell, label='skitscat', color='black', s=50, marker='v')
        
    title_text = '1 mio notional, historical ' + str(settings.HOLDING_PERIOD_IN_DAYS) + ' days basket return \n' 
    title_text += 'Trade entry with z score thresholds above +/-' + str(z_score_threshold)
    plt.title(title_text)

    plt.show()
    
# ---------------------------------------------------------------------------
# x_lim_tuple contains (start_date, end_date)
def plot_two_time_series_with_limits(time_series, x_lim_tuple, y_lim_tuple, vertline_date, num_days = settings.HOLDING_PERIOD_IN_DAYS):
   
    column_names = list(time_series.columns)
    
    if len(column_names) > 2:
        print('More than 2 columns, this function does not support.')
   
    ax = time_series.plot(secondary_y=[column_names[1]], kind='line', grid=True, figsize=(15,5), style='-o')
    
    plt.axvline(vertline_date, color='magenta', linestyle='--')
    
    # set the x limits for plotting
    start_date = x_lim_tuple[0]
    end_date = x_lim_tuple[1]    
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # set the lable for LHS y-axis
    ax.set_ylabel(column_names[0])
    
    # set the y limits for plotting
    ax.set_ylim(y_lim_tuple[0], y_lim_tuple[1])    
    
    title_text = 'For 1m Notional, '
    title_text += 'X = ' + str(num_days) + ' \n'   
    
    for name in column_names:
        val = time_series[name].iloc[-1]
        pnl = round(np.abs(settings.ONE_MILLION * val), 0)
        
        title_text += name + ' pnl equiv = USD ' + str(pnl) + ' \n'

    plt.title(title_text)


    plt.show()
    

    
# ---------------------------------------------------------------------------
# x_lim_tuple contains (start_date, end_date)
def plot_trading_signal_graphs(time_series_basket, basket_mean, basket_std, df_buy_sell_signal, x_lim_tuple, y_lim_tuple, vertline_date):
     
    ax = time_series_basket.plot(kind='line', grid=False, figsize=(15, 3), style='blue')
    
    # set the x limits for plotting
    start_date = x_lim_tuple[0]
    end_date = x_lim_tuple[1]    
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # set the y limits for plotting
    ax.set_ylim(y_lim_tuple[0], y_lim_tuple[1])        
    ax.set_ylabel("FX basket in USD points")

    # get the mean reversion level and standard deviation
    current_basket = time_series_basket.iloc[-1]
    current_basket_zscore = (current_basket - basket_mean) / basket_std
    
    #-------------------------------------------------------------------------  
    TODAY = time_series_basket.index[-1].strftime('%Y-%m-%d')
    
    dates_to_buy, retns_to_buy, dates_to_sell, retns_to_sell = ct.extract_data_conditions_on_buy_sell_signal(time_series_basket, 
                                                                                                             df_buy_sell_signal, 
                                                                                                             TODAY)
    
    if len(dates_to_buy) > 1:
        # plt.scatter dos not like if the array only has one element. That's why doing this check
        plt.scatter(dates_to_buy, retns_to_buy, label='skitscat', color='green', s=50, marker='^')
        
        
    if len(dates_to_sell) > 1:
        # plt.scatter dos not like if the array only has one element. That's why doing this check
        plt.scatter(dates_to_sell, retns_to_sell, label='skitscat', color='black', s=50, marker='v')   
     
     
    #-------------------------------------------------------------------------
    # plot the z score lines 
    mean_p_3std = basket_mean + 3.0 * basket_std
    mean_p_2std = basket_mean + 2.0 * basket_std
    mean_p_1std = basket_mean + 1.0 * basket_std
    mean_m_1std = basket_mean - 1.0 * basket_std
    mean_m_2std = basket_mean - 2.0 * basket_std
    mean_m_3std = basket_mean - 3.0 * basket_std
        
    plt.axhline(mean_p_3std, color='magenta', linestyle='--')
    plt.axhline(mean_p_2std, color='red', linestyle='--')
    plt.axhline(mean_p_1std, color='orange', linestyle='--')
    plt.axhline(basket_mean, color='black')  
    plt.axhline(mean_m_1std, color='lime', linestyle='--')
    plt.axhline(mean_m_2std, color='olive', linestyle='--')
    plt.axhline(mean_m_3std, color='green', linestyle='--')
    
    # vertical line for ease of looking
    plt.axvline(vertline_date, color='magenta', linestyle='--')    
    
    plt.legend(['basket:' + str(round(current_basket, 4)),                 
                '+3 std:' + str(round(mean_p_3std, 4)),   
                '+2 std:' + str(round(mean_p_2std, 4)),  
                '+1 std:' + str(round(mean_p_1std, 4)), 
                '  mean:' + str(round(basket_mean, 4)),                 
                '-1 std:' + str(round(mean_m_1std, 4)),  
                '-2 std:' + str(round(mean_m_2std, 4)),   
                '-3 std:' + str(round(mean_m_3std, 4))]);
                
    one_mio_one_sd_pnl_in_usd = round(np.abs(settings.ONE_MILLION * basket_std), 0)
    
    title_string = 'USD points of 1 SD = ' + str(round(basket_std, 4) ) + ' \n'
    title_string += '1 mio , 1 SD PnL in USD = $' + str(one_mio_one_sd_pnl_in_usd) + ' \n'
    title_string += 'Current Basket = ' + str(round(current_basket, 4) ) + ' \n'
    title_string += 'Current Z-Score = ' + str(round(current_basket_zscore, 4) ) + ' '   
    
    plt.title(title_string )

    plt.show()

# ------------------------------------------------------------------------------
def plot_johansen_stability_test(data, x_lim_tuple):
    start_date = x_lim_tuple[0]
    end_date = x_lim_tuple[1]  
    
    num_rows = data.shape[1]
    fig, axes = plt.subplots(nrows=num_rows, ncols=1, figsize=(15, 18))
    i = 0
    for column_name in list(data):
        data[column_name].plot(ax=axes[i], grid=True);
        
        last_value = data[column_name].iloc[-1]
        axes[i].set_title(column_name)
        axes[i].set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        i += 1  
    
# ------------------------------------------------------------------------
# back_test_res_df from btest.back_test_many_trades 
def plot_back_testing_results(back_test_res_df, SD_threshold, Sharpe_threshold, x_label, start_date, end_date):

    res_df, my_trade_df = btest.one_back_test_summary_table(back_test_res_df, 
                                                            start_date, 
                                                            end_date, 
                                                            Sharpe_threshold, 
                                                            SD_threshold)

    print(res_df)

    if x_label == 'Sharpe Ratio':
        x_axis = 'Sharpe Ratio on Trade Date' 
    elif x_label == 'Abs SD':
        x_axis = 'Abs Z Score on Trade Date' 
        
    y_axisl = '5D Realized Rtns in SD' 
    y_axis2 = '5D max DD in SD'

    sns.lmplot(x_axis, y_axisl, hue='Trade Date', data=my_trade_df, fit_reg=False, size=4, aspect=2) 
    sns.lmplot(x_axis, y_axis2, hue='Trade Date', data=my_trade_df, fit_reg=False, size=4, aspect=2)

# ------------------------------------------------------------------------
# x_lim_tuple contains (start_date, end_date) 
def plot_is_still_coint_graphs(time_series_basket, basket_mean, basket_std, df_still_coint, x_lim_tuple, y_lim_tuple, vertline_date):

    ax = time_series_basket.plot(kind='line', grid=True, figsize=(15, 3), style='blue')

    # set the x limits for plotting
    start_date = x_lim_tuple[0]
    end_date = x_lim_tuple[1] 
    ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))

    # set the y limits for plotting 
    ax.set_ylim(y_lim_tuple[0], y_lim_tuple[1]) 
    ax.set_ylabel("FX basket in USD points")

    # get the mean reversion level and standard deviation
    current_basket = time_series_basket.iloc[-1] 
    current_basket_zscore = (current_basket - basket_mean) / basket_std

    # find out the dates which are not cointegrated 
    df_trace_10_test_fail = df_still_coint[~df_still_coint['trace (10%)']]    
    dates_coint_fail = df_trace_10_test_fail.index 
    basket_values_at_fail_dates = time_series_basket[dates_coint_fail]
    
    df_trace_10_test_pass = df_still_coint[df_still_coint['trace (10%)']]    
    dates_coint_pass = df_trace_10_test_pass.index 
    basket_values_at_pass_dates = time_series_basket[dates_coint_pass]

    if len(dates_coint_fail) > 1: 
        # plt.scatter does not like if the array only has one element. That's why doing this check 
        plt.scatter(dates_coint_fail, basket_values_at_fail_dates, label='skitscat', color='red', s=50, marker="x")
        
    if len(dates_coint_pass) > 1: 
        # plt.scatter does not like if the array only has one element. That's why doing this check 
        plt.scatter(dates_coint_pass, basket_values_at_pass_dates, label='skitscat', color='green', s=50, marker="x")

    # vertical line for ease of looking 
    plt.axvline(vertline_date, color='magenta', linestyle='--')

    plt.legend(['basket:' + str(round(current_basket, 4)), 
                '  mean:' + str(round(basket_mean, 4)),
                ]) 
                
    title_string = 'Red Cross means the basket is NOT cointegrated after the Trade Date (verticle line) \n' 
    
    plt.title(title_string )

    plt.show()