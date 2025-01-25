import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import datetime
import data_io as myio
import model_settings as settings
import copy

# ------------------------------------------------------------------ 
def time_now():
    return datetime.datetime.now()
    
# ---------------------------------------------------------------
def zscore(level, mean, std):
    res = (level - mean) / std
    return res
    
# ------------------------------------------------------------------
# assume the input data has daily frequency
# basket - the time series which we use to estimate the daily vol
def compute_daily_hist_normal_vol(basket, period=15):

    # compute daily change
    basket_diff = basket.diff()      
    
    # compute the stdev of the daily change
    diff_rolling_std_daily = basket_diff.rolling(period).std()        
    
    diff_rolling_std_daily = diff_rolling_std_daily.rename('daily stdev with ' + str(period) + ' periods')
    
    return diff_rolling_std_daily

# -----------------------------------------------------------------
# This is a faster way to compute the last normal vol, rather than compute all and take the last one
def compute_last_daily_hist_normal_vol(basket, period=settings.NUM_PERIOD_FOR_HIST_VOL_EST):

    sub_set_basket = basket.iloc[-(period+1):]
    
    basket_diff = sub_set_basket.diff()
    
    std_daily = basket_diff.std()
    
    return std_daily

# ------------------------------------------------------------------    
# Given a time series of fx spot and the weights. Compute a time series of the basket
def compute_basket(df_fx_name_list_XXXUSD, weights_XXXUSD):    
    basket = pd.Series(np.dot(df_fx_name_list_XXXUSD, weights_XXXUSD), name='Basket')        
    basket.index = df_fx_name_list_XXXUSD.index    
    return basket

  
# --------------------------------------------------------------- 
def create_position_df(data_XXXUSD, weights_XXXUSD):        
    name_list = list(data_XXXUSD.keys())
    XXXUSD_last = data_XXXUSD.iloc[-1].values    
    
    N = len(name_list)
    
    # initialize 
    weights_market_convention = np.array(weights_XXXUSD)
    FX_market_convention = np.array(XXXUSD_last)

    # compute the weights in market convention quoting
    for x in range(N):    
        if myio.is_inverted_quote(name_list[x]):
            weights_market_convention[x] = -XXXUSD_last[x]*weights_XXXUSD[x]   
            FX_market_convention[x] = 1.0/FX_market_convention[x]
     
    #-------------------------------------------------------------------------    
    USD_amount = 0
    
    # compute USD amount
    for x in range(N):   
        
        if myio.is_inverted_quote(name_list[x]):      
            # like SGDUSD or CNHUSD
            USD_amount += weights_market_convention[x]                    
        else:       
            # like GBPUSD, AUDUSD
            USD_amount += -weights_XXXUSD[x] * FX_market_convention[x]

    #-------------------------------------------------------------------------    
    res_df = pd.DataFrame(list(zip(weights_XXXUSD, FX_market_convention, weights_market_convention)), 
                          index = name_list, 
                          columns=['weights', 'PX_LAST', 'market convention notional'] )

    return res_df, USD_amount      

# ---------------------------------------------------------------     
# data_XXXUSD - data contains the fx rates of the basket in XXXUSD
# notionals - array contains the notionals of 1st currency pair in the basket
# weights_XXXUSD - weights of the basket for XXXUSD
# traded_fx_rates_in_XXXUSD - traded fx rates in format of XXXUSD
# start_date  in the format of yyyymmdd
# end_date  in the format of yyyymmdd
def pnl_analysis_info_to_Excel(data_XXXUSD, 
                               notionals,
                               weights_XXXUSD, 
                               traded_fx_rates_in_XXXUSD, 
                               start_date, 
                               end_date):        

    name_list = data_XXXUSD.keys()
    
    # tenor product to get the weights of the basket for every time slice XXXUSD
    notionals_weights = np.tensordot(notionals, weights_XXXUSD, axes=0)    
    
    # getting the dates (from start_date to end_date) as row label aka index.
    date_index = data_XXXUSD.loc[start_date:end_date].index    
    
    # create a dataframe for notional times weights indexed by date 
    notional_weights_df = pd.DataFrame(notionals_weights, index=date_index, columns=name_list )
     
    # create a series for notional indexed by date 
    notionals_df = pd.Series(list(notionals), index=date_index, name='notional' )   
    
    sd = pd.to_datetime(start_date, format='%Y-%m-%d') 
    ed = pd.to_datetime(end_date, format='%Y-%m-%d') 
    
    # the list to store the culmulative pnls for basket and daily pnls
    all_pnls = []
    
    basket_cum_pnl = 0
    
    n_prev = notionals_df.loc[sd]
    w_prev = notional_weights_df.loc[sd]
    fx_prev = traded_fx_rates_in_XXXUSD    
    
    # loop over each date (this loop can be optimized)
    for t in data_XXXUSD.index:  

        if t <= ed and t > sd:        
            fx_t = data_XXXUSD.loc[t]    
            
            # daily pnl = w_0 *(fx_1 - fx_0), we assume constant notional for 1 day.
            basket_daily_pnl = np.dot(w_prev, fx_t - fx_prev)            
            basket_cum_pnl += basket_daily_pnl    
            
            # list comprehension on 3 lists in order to compute: w*(fx(t)-fx(0)). Essentially a single for-loop over 3 arrays.
            fx_cum_pnls = []
            [fx_cum_pnls.append( w * (p1 - p0)) for p1, p0, w in zip(fx_t, fx_prev, w_prev) ] 

            # merging 2 tuples and then append to all_pnls
            all_pnls.append((t, n_prev, basket_daily_pnl, basket_cum_pnl) + tuple(fx_cum_pnls))

            # reset the variable for next iteration, except for the last slice
            if t != ed:
                n_prev = notionals_df.loc[t]
                w_prev = notional_weights_df.loc[t]
                fx_prev = fx_t     
    
    name_list = ['Dates', 'Notional', 'Basket Daily PnL', 'Basket Cumulative PnL'] + list(data_XXXUSD.keys())
    
    pnls_df = pd.DataFrame(all_pnls, columns=name_list)    
        
    pnls_df = pnls_df.set_index('Dates')
    
    return pnls_df
    
# --------------------------------------------------------------------------------
# compute the x number of day change histogram by upper and lower bound of z-score
# time series to be used to compute the change
# lower_zs - lower_z-score
# upper_zs - upper z-score
# numday_return in integer, 5 = 5 days
def conditional_numday_change_hist(time_series, lower_zs, upper_zs, numday_return):
    
    # compute the daily change fro t to t+1
    temp = time_series.diff(numday_return)
    
    # shift back the days so that they are aligned
    temp = temp.shift(-numday_return)
    
    # remove the NA
    temp = temp.iloc[:-numday_return]
    
    # create the x day change time series
    ds = pd.Series(temp, name='x daily change')
    
    # compute the z score at t and make the dimension to be the same as 'x daily change'
    df_zscore = pd.Series(zscore(time_series).iloc[:-numday_return], name='z score')   
           
    df_filter = pd.concat([ds, df_zscore], axis=1)
    
    # set up the filtering conditions
    low_condition = df_filter['z score'] > lower_zs    
    up_condition = df_filter['z score'] < upper_zs
    
    # do the filtering
    df_filter = df_filter[low_condition]
    df_filter = df_filter[up_condition]
    
    cond_ts = df_filter['x daily change']
    mean = cond_ts.mean()
    std = cond_ts.std()
    
    plt.figure(figsize=(20,10)) 
    ax = cond_ts.hist(bins=100)
    ax.set_title(str(numday_return) + ' day change. \n' + 'Mean = ' + str(round(mean,4)) + '. Std = ' + str(round(std,4) ) )
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Dates')
    
    plt.axvline(mean+std, color='red', linestyle='--')
    plt.axvline(mean, color='yellow', linestyle='--')
    plt.axvline(mean-std, color='green', linestyle='--')



# --------------------------------------------------------------------
# compute the min and max change for a given range of z_score
def conditional_min_max_change(basket, z_score_time_series, num_days, zscore_range_tuple):    

    # compute the abs x days return
    col_name = str(num_days)+' Day Change'
    diff = pd.Series(basket.diff(periods=num_days), name=col_name) 
    
    # shift back the days so that z core and diff series are aligned
    diff = diff.shift(-num_days)
    
    # combine to form a dataframe
    df_filter = pd.concat([diff, z_score_time_series], axis=1)
    
    # remove the rows which x day change is NA 
    df_filter = df_filter.iloc[:-num_days]
    
    # set up the filtering conditions
    lower_zs = zscore_range_tuple[0]
    upper_zs = zscore_range_tuple[1]       
    
    low_condition = df_filter['z score'] > lower_zs    
    up_condition = df_filter['z score'] < upper_zs
    
    print('--------------------------------------------------------')
    print('Between Z-Score ' + str(lower_zs) + ' and ' + str(upper_zs) )
    
    # do the filtering
    df_filter = df_filter[low_condition]
    df_filter = df_filter[up_condition]
    
    # get the xd change column only
    filtered_xd_change = df_filter[col_name]
    
    notional = settings.ONE_MILLION

    # min change 
    min_change = filtered_xd_change.min(axis=0)    
    idx_min = filtered_xd_change.idxmin(axis=0).strftime('%Y-%m-%d')        
    print('For 1mio notional, min ' + str(num_days) + ' day PnL = ' + str(round(notional * min_change, 2)) + ' at ' + str(idx_min) )
    
    # max change     
    max_change = filtered_xd_change.max(axis=0)    
    idx_max = filtered_xd_change.idxmax(axis=0).strftime('%Y-%m-%d')    
    print('For 1mio notional, max ' + str(num_days) + ' day PnL = ' + str(round(notional * max_change, 2)) + ' at ' + str(idx_max) )

    
#-------------------------------------------------------------------------------------------
def compute_x_day_historical_returns(basket, time_in_days=settings.HOLDING_PERIOD_IN_DAYS):
    
    # compute realized x day basket return
    basket_x_day_return = basket.diff(time_in_days)

    # shift the index backward to align with the predcition date
    basket_x_day_return = basket_x_day_return.shift(-time_in_days)
    
    basket_x_day_return = basket_x_day_return.rename('x days basket return')

    return basket_x_day_return

# -----------------------------------------------------------------------------------------------
# Note that the mean cancels out if we take diff. So we don't need the mean.
def compute_x_day_historical_returns_in_SD(basket, time_in_days, basket_stdev):
    res = compute_x_day_historical_returns(basket, time_in_days)    
    res = res / basket_stdev    
    return res
    
# -----------------------------------------------------------------------------------------------    
def compute_pct_change(time_series, num_of_days, to_shift):    

    name_tag = time_series.name + ' pctChange'
    
    # initialize the dataframe
    df = pd.DataFrame()

    # compute the difference for x days to 1 day
    for i in range(1, num_of_days+1):
        pctChange = time_series.pct_change(periods=i)

        if to_shift == settings.ToShiftOrNot.TO_SHIFT: 
            # shift so that it is corresponding to the trade date
            pctChange = pctChange.shift(-i)

        # give a unique name for each difference
        pctChange.name = name_tag + str(i)

        # concat to the same dataframe
        df = pd.concat([df, pctChange], axis = 1)
        
    return df

# -----------------------------------------------------------------------------------------------    
def compute_diff(time_series, num_of_days, to_shift):    

    name_tag = time_series.name + ' diff'
    
    # initialize the dataframe
    df = pd.DataFrame()

    # compute the difference for x days to 1 day
    for i in range(1, num_of_days+1):
        diff = time_series.diff(i)

        if to_shift == settings.ToShiftOrNot.TO_SHIFT: 
            # shift so that it is corresponding to the trade date
            diff = diff.shift(-i)

        # give a unique name for each difference
        diff.name = name_tag + str(i)

        # concat to the same dataframe
        df = pd.concat([df, diff], axis = 1)
        
    return df
    
# -----------------------------------------------------------------------------------------
# compute the min and max loss over x days of a basket over a period of trade dates
def min_max_return_x_days(basket, time_in_days):

    # initialize the dataframe
    df = compute_diff(basket, time_in_days, settings.ToShiftOrNot.TO_SHIFT)
        
    # compute the min and max across the columns
    diff_max = df.max(axis=1)
    diff_max.name = 'max'
    
    diff_min = df.min(axis=1)
    diff_min.name = 'min'
    
    # concate to the dataframe
    df = pd.concat([diff_max, diff_min, df], axis=1)
    
    df = df.dropna()
    
    return df
    
# -------------------------------------------------------------------------------------------------
def min_max_return_x_days_in_SD(basket, time_in_days, basket_stdev):
    res = min_max_return_x_days(basket, time_in_days)
    res = res / basket_stdev
    return res
    
# start and until are in the format of '2020-06-15'
def generate_Tue_Thur_between_2_dates(start_str, until_str):

    start_date = datetime.datetime.strptime(start_str, '%Y-%m-%d')
    until_date = datetime.datetime.strptime(until_str, '%Y-%m-%d')
    
    num_days = (until_date-start_date).days
    
    result_days = []
    
    for d in range(1, num_days+1):
        res_date = start_date + datetime.timedelta(days=d)
        
        # Tuesday = 2, Thursday = 4
        if res_date.isoweekday() == 2 or res_date.isoweekday() == 4:
            result_days.append(res_date.strftime('%Y-%m-%d'))
            
    return result_days
    
# start and until are in the format of '2020-06-15'
def generate_Mon_to_Fri_between_2_dates(start_str, until_str):

    start_date = datetime.datetime.strptime(start_str, '%Y-%m-%d')
    until_date = datetime.datetime.strptime(until_str, '%Y-%m-%d')
    
    num_days = (until_date-start_date).days
    
    result_days = []
    
    for d in range(1, num_days+1):
        res_date = start_date + datetime.timedelta(days=d)
        
        # Tuesday = 2, Thursday = 4
        if res_date.isoweekday() >= 1 and res_date.isoweekday() <= 5:
            result_days.append(res_date.strftime('%Y-%m-%d'))
            
    return result_days
    
# Pip size definition for each currency pair in market convention format 
def pip_size(key):    

    bbg_dict = {'EURUSD Curncy': 0.0001, 
                'GBPUSD Curncy': 0.0001, 
                'AUDUSD Curncy': 0.0001, 
                'NZDUSD Curncy': 0.0001,             
                'JPYUSD Curncy': 0.01, 
                'CADUSD Curncy': 0.0001, 
                'CHFUSD Curncy': 0.0001, 
                'NOKUSD Curncy': 0.0001, 
                'SEKUSD Curncy': 0.0001,    
                'SGDUSD Curncy': 0.0001, 
                'CNHUSD Curncy': 0.0001      
                }
            
    return bbg_dict.get(key)   
    
# compute the change in basket level when each currency pairs pertubed by x pips
def basket_level_impact_due_to_fx_spot_change(df_fx_name_list_XXXUSD, weights_XXXUSD, shock_in_market_conv_pips):
    
    res = []    
    name_list = []
    
    for col in df_fx_name_list_XXXUSD.columns:    

        # deep copy of the PX LAST for all fx pairs
        ALL_PX_LAST = copy.deepcopy(df_fx_name_list_XXXUSD.iloc[-1])
    
        # get one FX pair
        One_FX_PX_LAST = ALL_PX_LAST.loc[col]
        
        shock = shock_in_market_conv_pips * pip_size(col)
    
        if myio.is_inverted_quote(col):
            temp = 1.0/One_FX_PX_LAST + shock
            shocked_PX_LAST = 1.0/temp
        else:
            shocked_PX_LAST = One_FX_PX_LAST + shock
        
        # apply the shock and replace the PX LAST
        ALL_PX_LAST.loc[col] = shocked_PX_LAST
        
        # compute the basket using the shocked PX LAST
        res.append(np.dot(ALL_PX_LAST, weights_XXXUSD))       
        
    res_series = pd.Series(res)        
    res_series.index = df_fx_name_list_XXXUSD.columns
    
    return res_series

# compute the change in basket SD when each currency pairs pertubed by x pips
def basket_SD_change_due_to_fx_spot_change(df_fx_name_list_XXXUSD, weights_XXXUSD, shock_in_market_conv_pips, long_run_mean, basket_stdev):    
    
    oneMioOneSD = basket_stdev * settings.ONE_MILLION   
    
    # series of new basket level due to individual fx spot shocked
    res_level = basket_level_impact_due_to_fx_spot_change(df_fx_name_list_XXXUSD, weights_XXXUSD, shock_in_market_conv_pips)  

    # compute the new SD level  
    res_SD = (res_level - long_run_mean) / basket_stdev
    
    # get the current SD level
    current_SD = (np.dot(df_fx_name_list_XXXUSD.iloc[-1], weights_XXXUSD) - long_run_mean) / basket_stdev
    
    res_delta_SD = []
    SD_name_list = []
    delta_name_list = []
    
    # compute the SD delta
    for col in df_fx_name_list_XXXUSD.columns: 
        res_delta_SD.append( (res_SD.loc[col] - current_SD) * oneMioOneSD)
        
        SD_name_list.append(col + ' Basket SD after ' + str(shock_in_market_conv_pips) + ' pips shock')
        delta_name_list.append(col + str(' 1mio 1SD USD delta'))
    
    # reset the index for the Basket SD Series
    res_SD.index = SD_name_list
    
    res_delta_SD_series = pd.Series(res_delta_SD)
    res_delta_SD_series.index = delta_name_list  
    
    # join 2 series together for ease of display
    res_SD = pd.concat([res_SD,res_delta_SD_series])
    
    return res_SD
    
    

