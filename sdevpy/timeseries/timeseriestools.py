import numpy as np
import pandas as pd
from enum import Enum


class ToShiftOrNot(Enum):
    TO_SHIFT = 1
    NOT_SHIFT = 0


# [compute_diff]
def compute_diff(time_series, num_of_days, to_shift):
    name_tag = time_series.name + ' diff'

    # Initialize the dataframe
    df = pd.DataFrame()

    # Compute the difference for x days to 1 day
    for i in range(1, num_of_days+1):
        diff = time_series.diff(i)

        if to_shift == ToShiftOrNot.TO_SHIFT:
            # Shift so that it is corresponding to the trade date
            diff = diff.shift(-i)

        # Give a unique name for each difference
        diff.name = name_tag + str(i)

        # Concat to the same dataframe
        df = pd.concat([df, diff], axis = 1)

    return df


# [min_max_return_x_days]
def min_max_return_x_days(basket, time_in_days):
    """ Compute the min and max loss over x days of a basket over a period of trade dates """
    # Initialize the dataframe
    df = compute_diff(basket, time_in_days, ToShiftOrNot.TO_SHIFT)

    # Compute the min and max across the columns
    diff_max = df.max(axis=1)
    diff_max.name = 'max'

    diff_min = df.min(axis=1)
    diff_min.name = 'min'

    # Concate to the dataframe
    df = pd.concat([diff_max, diff_min, df], axis=1)
    df = df.dropna()

    return df


# [min_max_return_x_days_in_SD]
def min_max_return_x_days_in_SD(basket, time_in_days, basket_stdev):
    res = min_max_return_x_days(basket, time_in_days)
    return res / basket_stdev


# [compute_x_day_historical_returns_in_SD]
def x_day_historical_returns_in_SD(basket, time, basket_stdev):
    """ Note that the mean cancels out if we take diff. So we don't need the mean """
    res = x_day_historical_returns(basket, time)
    return res / basket_stdev


def x_day_historical_returns(basket, time_in_days=5):
    # Compute realized x day basket return
    basket_x_day_return = basket.diff(time_in_days)

    # Shift the index backward to align with the predcition date
    basket_x_day_return = basket_x_day_return.shift(-time_in_days)
    basket_x_day_return = basket_x_day_return.rename('x days basket return')

    return basket_x_day_return


def is_inverted_quote(key):
    is_inverted = False
    bbg_dict = {'EURUSD Curncy': False, 'GBPUSD Curncy': False, 'AUDUSD Curncy': False,
                'NZDUSD Curncy': False, 'JPYUSD Curncy': True, 'CADUSD Curncy': True,
                'CHFUSD Curncy': True, 'NOKUSD Curncy': True, 'SEKUSD Curncy': True,
                'SGDUSD Curncy': True, 'CNHUSD Curncy': True}

    is_inverted = bbg_dict.get(key)
    return is_inverted

# [compute_basket]
def weighted_series(df_data, weights):
    """ Given time series of fx spots and the weights, compute the dot product time series """
    series = pd.Series(np.dot(df_data, weights), name='Series')
    series.index = df_data.index
    return series

# [create_position_df]
def create_position(df_data, weights):
    name_list = list(df_data.keys())
    XXXUSD_last = df_data.iloc[-1].values
    N = len(name_list)

    # Initialize
    weights_market_convention = np.array(weights)
    FX_market_convention = np.array(XXXUSD_last)

    # Compute the weights in market convention quoting
    for x in range(N):
        if is_inverted_quote(name_list[x]):
            weights_market_convention[x] = -XXXUSD_last[x] * weights[x]
            FX_market_convention[x] = 1.0 / FX_market_convention[x]

    # Compute USD amount
    USD_amount = 0
    for x in range(N):
        if is_inverted_quote(name_list[x]): # Like SGDUSD or CNHUSD
            USD_amount += weights_market_convention[x]
        else: # Like GBPUSD, AUDUSD
            USD_amount += -weights[x] * FX_market_convention[x]

    res_df = pd.DataFrame(list(zip(weights, FX_market_convention, weights_market_convention)), 
                          index = name_list, 
                          columns=['weights', 'PX_LAST', 'market convention notional'])

    return res_df, USD_amount


def last_daily_hist_normal_vol(series, period=15):
    subset_series = series.iloc[-(period + 1):]
    returns = subset_series.diff()
    return returns.std()


def daily_hist_normal_vol(series, period=15):
    """ Assume the input data has daily frequency """
    # Compute daily changes
    returns = series.diff()

    # Compute the stdev of the daily change
    diff_rolling = returns.rolling(period).std()
    return diff_rolling.rename('daily stdev with ' + str(period) + ' periods')
