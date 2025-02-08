import numpy as np
import pandas as pd


def is_inverted_quote(key):
    is_inverted = False
    bbg_dict = {'EURUSD Curncy': False, 'GBPUSD Curncy': False, 'AUDUSD Curncy': False,
                'NZDUSD Curncy': False, 'JPYUSD Curncy': True, 'CADUSD Curncy': True,
                'CHFUSD Curncy': True, 'NOKUSD Curncy': True, 'SEKUSD Curncy': True,
                'SGDUSD Curncy': True, 'CNHUSD Curncy': True}

    is_inverted = bbg_dict.get(key)
    return is_inverted


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
