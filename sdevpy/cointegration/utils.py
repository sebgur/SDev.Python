import numpy as np
import pandas as pd
import datetime
from sdevpy.cointegration import data_io as myio
from sdevpy.cointegration import model_settings as settings


def time_now():
    return datetime.datetime.now()


def zscore(level, mean, std):
    res = (level - mean) / std
    return res


def compute_daily_hist_normal_vol(basket, period=15):
    """ Assume the input data has daily frequency basket
        - the time series which we use to estimate the daily vol """
    # compute daily change
    basket_diff = basket.diff()

    # compute the stdev of the daily change
    diff_rolling_std_daily = basket_diff.rolling(period).std()

    diff_rolling_std_daily = diff_rolling_std_daily.rename('daily stdev with ' + str(period) + ' periods')

    return diff_rolling_std_daily


def compute_last_daily_hist_normal_vol(basket, period=settings.NUM_PERIOD_FOR_HIST_VOL_EST):
    """ This is a faster way to compute the last normal vol, rather than compute all and take the last one """
    sub_set_basket = basket.iloc[-(period+1):]
    basket_diff = sub_set_basket.diff()
    std_daily = basket_diff.std()
    return std_daily


def compute_basket(df_fx_name_list_xxxusd, weights_xxxusd):
    """ Given a time series of fx spot and the weights. Compute a time series of the basket """
    basket = pd.Series(np.dot(df_fx_name_list_xxxusd, weights_xxxusd), name='Basket')
    basket.index = df_fx_name_list_xxxusd.index
    return basket


def create_position_df(data_xxxusd, weights_xxxusd):
    name_list = list(data_xxxusd.keys())
    xxxusd_last = data_xxxusd.iloc[-1].values
    n = len(name_list)

    # Initialize
    weights_market_convention = np.array(weights_xxxusd)
    fx_market_convention = np.array(xxxusd_last)

    # Compute the weights in market convention quoting
    for x in range(n):
        if myio.is_inverted_quote(name_list[x]):
            weights_market_convention[x] = -xxxusd_last[x] * weights_xxxusd[x]
            fx_market_convention[x] = 1.0 / fx_market_convention[x]

    usd_amount = 0

    # Compute USD amount
    for x in range(n):
        if myio.is_inverted_quote(name_list[x]):
            # like SGDUSD or CNHUSD
            usd_amount += weights_market_convention[x]
        else:
            # like GBPUSD, AUDUSD
            usd_amount += -weights_xxxusd[x] * fx_market_convention[x]

    res_df = pd.DataFrame(list(zip(weights_xxxusd, fx_market_convention, weights_market_convention, strict=True)),
                          index = name_list,
                          columns=['weights', 'PX_LAST', 'market convention notional'] )

    return res_df, usd_amount


def compute_x_day_historical_returns(basket, time_in_days=settings.HOLDING_PERIOD_IN_DAYS):
    # compute realized x day basket return
    basket_x_day_return = basket.diff(time_in_days)

    # shift the index backward to align with the predcition date
    basket_x_day_return = basket_x_day_return.shift(-time_in_days)
    basket_x_day_return = basket_x_day_return.rename('x days basket return')

    return basket_x_day_return


def compute_x_day_historical_returns_in_sd(basket, time_in_days, basket_stdev):
    """ Note that the mean cancels out if we take diff. So we don't need the mean. """
    res = compute_x_day_historical_returns(basket, time_in_days)
    res = res / basket_stdev
    return res


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


def min_max_return_x_days(basket, time_in_days):
    """ Compute the min and max loss over x days of a basket over a period of trade dates """
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


def min_max_return_x_days_in_sd(basket, time_in_days, basket_stdev):
    res = min_max_return_x_days(basket, time_in_days)
    res = res / basket_stdev
    return res
