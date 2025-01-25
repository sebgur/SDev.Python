import os
import unittest
import mean_reversion as my_mean_rev 
import data_io as myio
import utils as ut
import pandas as pd
import coint_trading as ct
import back_testing as btest
import math
import model_settings as settings


# ----Steps to run this unit test 
# (1) In Condo command promot, go to the directory of this file 
# (2) type python run_unit_test.py

ROOT = r"C:\\temp\\sdevpy\\cointegration"


class Test_mean_reversion(unittest.TestCase):
    # mean_rev_expected_and_variance_change
    def test_1(self): 
        mean_rev_level = -0.2144341010367362 
        mean_rev_rate_in_days = -0.04433266211937964 
        time_in_days = 5.0 
        X0 = -0.12860286528510775 
        daily_hist_normal_vol = 0.00963634066198962

        EdX, vardX = my_mean_rev.mean_rev_expected_and_variance_change(mean_rev_level, 
                                                                       mean_rev_rate_in_days,
                                                                       time_in_days, 
                                                                       X0,
                                                                       daily_hist_normal_vol)

        self.assertEqual(EdX, -0.017064531174852654)
        self.assertEqual(vardX, 0.0003750401959678947)

    # compute_buy_sell_signa1 
    def test_2(self):
        abs_lower_threshold = 2 
        abs_higher_threshold = 9

        # No buy and sell
        current_zscore = 1.8888
        buy_signal, sell_signal = my_mean_rev.compute_buy_sell_signal(current_zscore, abs_lower_threshold, abs_higher_threshold)
        self.assertFalse(buy_signal)
        self.assertFalse(sell_signal)

        # Sell 
        buy_signal, sell_signal = my_mean_rev.compute_buy_sell_signal(2.3, abs_lower_threshold, abs_higher_threshold) 
        self.assertFalse(buy_signal) 
        self.assertTrue(sell_signal)

        # Buy 
        buy_signal, sell_signal = my_mean_rev.compute_buy_sell_signal(-2.3, abs_lower_threshold, abs_higher_threshold) 
        self.assertTrue(buy_signal) 
        self.assertFalse(sell_signal)

        # No buy and sell 
        buy_signal, sell_signal = my_mean_rev.compute_buy_sell_signal(4, abs_lower_threshold, 3)
        self.assertFalse(buy_signal) 
        self.assertFalse(sell_signal)

        # No buy and sell 
        buy_signal, sell_signa1 = my_mean_rev.compute_buy_sell_signal(-4, abs_lower_threshold, 3) 
        self.assertFalse(buy_signal) 
        self.assertFalse(sell_signal)

    # compute_mean_reversion_params
    def test_3(self):
        res_dict = my_mean_rev.compute_mean_reversion_params(basket)

        self.assertAlmostEqual(res_dict['Half Life in days'], 17.86467097218617, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res_dict['Mean Rev Rate in days'], -0.038799885071441775, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res_dict['Mean Rev Level'], 0.533232617056516, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res_dict['const p-value'], 6.983194829903289e-07, 13, "error bigger than 13 dp")
        self.assertAlmostEqual(res_dict['Basket p-value'], 6.264859941908779e-07, 13, "error bigger than 13 dp")


    # compute_sharpe_and_buysell_signal_multi_period 
    def test_4(self):
        lower_threshold = 1
        higher_threshold = 3

        res = my_mean_rev.compute_sharpe_and_buysell_signal_multi_period(basket,
                                                                         mean_rev_level, 
                                                                         basket_stdev, 
                                                                         mean_rev_rate_in_days, 
                                                                         settings.HOLDING_PERIOD_IN_DAYS,
                                                                         lower_threshold,
                                                                         higher_threshold)

        self.assertAlmostEqual(res['Sharpe Ratio'].loc['2020-06-02'], 0.11978127715928039, 13, "error bigger than 13 dp")  
        self.assertAlmostEqual(res['Return Expectation over X days'].loc['2020-06-02'], 0.004050850524515148, 13, "error bigger than 13 dp")  
        self.assertAlmostEqual(res['Return SD over X days'].loc['2020-06-02'], 0.03381872877451864, 13, "error bigger than 13 dp") 
        self.assertEqual(res['Buy Signal'].loc['2020-05-22'], 1) 
        self.assertEqual(res['Sell Signal'].loc['2020-02-17'], 1)

    # compute_sharpe_ratio 
    def test_5(self):
        mean_rev_level = -0.2144341010367362 
        mean_rev_rate_in_days = -0.04433266211937964 
        time_in_days = 5.0
        X0 = -0.12860286528510775
        daily_hist_normal_vol = 0.00963634066198962
        z_score = 1

        res = my_mean_rev.compute_sharpe_ratio(mean_rev_level, 
                                               mean_rev_rate_in_days, 
                                               time_in_days, 
                                               X0, 
                                               daily_hist_normal_vol, 
                                               z_score)

        self.assertAlmostEqual(res['Sharpe Ratio'], 0.8811613764334744, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res['Return Expectation'], 0.017064531174852654, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res['Return SD'], 0.01936595455865511, 13, "error bigger than 13 dp") 

    # class MeanRevTimeSeries and zscores_mean_revert_time_series 
    def test_6(self):
        z_score_ts = my_mean_rev.zscores_mean_revert_time_series(basket, mean_rev_level, basket_stdev) 
        self.assertAlmostEqual(z_score_ts.iloc[0], -0.5914504308234214, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(z_score_ts.iloc[-1],  -0.4510901317894131, 13, "error bigger than 13 dp") 

        # compute the z score time series using the class MeanRevTimeSeries, should be the same 
        my_ts = my_mean_rev.MeanRevTimeSeries(basket)
        z_score_ts2 = my_ts.get_zscores_time_series()

        self.assertEqual(z_score_ts.iloc[0], z_score_ts2.iloc[0]) 
        self.assertEqual(z_score_ts.iloc[5], z_score_ts2.iloc[5]) 
        self.assertEqual(z_score_ts.iloc[-5], z_score_ts2.iloc[-5]) 
        self.assertEqual(z_score_ts.iloc[-1], z_score_ts2.iloc[-1])


class Test_utils(unittest.TestCase):
    # create_position_df 
    def test_1(self):
        res_df, USD_amount = ut.create_position_df(df_fx_name_list_XXXUSD, weights_XXXUSD)
        
        self.assertAlmostEqual(USD_amount, -0.5102610096227571, 13, "error bigger than 13 dp") 

        dict = {'weights':[1.0, -1.64293648, 2.67290927, 1.29586726, -16.154956], 
                'PX_LAST': [1.2551, 0.6897, 0.6371, 1.3519, 7.106800000000001], 
                'market convention notional':[1.0, -1.64293648, 2.67290927, -0.9585526000443819, 2.273168796082625]}

        correct_res_df = pd.DataFrame(dict) 
        correct_res_df.index = name_list

        self.assertTrue(correct_res_df.equals(res_df))
        

    # compute_daily_hist_normal_vol 
    # compute_last_daily_hist_normal_vol 
    def test_2(self):
        normal_vol_ts = ut.compute_daily_hist_normal_vol(basket, 15)

        vol_at_2015_08_12 = normal_vol_ts.loc['2015-08-12'] 
        self.assertTrue(math.isnan(vol_at_2015_08_12))

        vol_at_2020_06_02 = normal_vol_ts.loc['2020-06-02'] 
        self.assertAlmostEqual(vol_at_2020_06_02, 0.0166126278779614, 13, "error bigger than 13 dp") 

        # vol_at_2020_06_02 and latest_vol are slightly different by lE-15. Not sure why
        latest_vol = ut.compute_last_daily_hist_normal_vol(basket, 15) 
        self.assertAlmostEqual(latest_vol, 0.01661262787796141, 13, "error bigger than 13 dp") 

    # compute_x_day_historica1_returns
    def test_3(self): 
        res = ut.compute_x_day_historical_returns(basket, 5) 
        self.assertAlmostEqual(res.loc['2020-05-26'], 0.033510398378947315, 13, "error bigger than 13 dp") 
        self.assertTrue(math.isnan(res.loc['2020-06-02']))

    # compute_x_day_historical_returns in SD
    def test_4(self):
        basket_stdev = 0.04 
        res = ut.compute_x_day_historical_returns_in_SD(basket, 5, basket_stdev)    
        self.assertEqual(res.loc['2020-05-26'], 0.8377599594736829) 
        self.assertTrue(math.isnan(res.loc['2020-06-02']))
        
    # basket_SD_change_due_to_fx_spot_change
    def test_5(self):
        shock_in_market_conv_pips = 10   
        res = ut.basket_SD_change_due_to_fx_spot_change(df_fx_name_list_XXXUSD, weights_XXXUSD, shock_in_market_conv_pips, mean_rev_level, basket_stdev)
        
        self.assertAlmostEqual(res.loc['GBPUSD Curncy Basket SD after 10 pips shock'], -0.4314532764631008, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res.loc['AUDUSD Curncy Basket SD after 10 pips shock'], -0.483352237757477, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res.loc['NZDUSD Curncy Basket SD after 10 pips shock'], -0.3986025991540779, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res.loc['CADUSD Curncy Basket SD after 10 pips shock'], -0.4650031768997174, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res.loc['CNHUSD Curncy Basket SD after 10 pips shock'], -0.4448100047766933, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res.loc['GBPUSD Curncy 1mio 1SD USD delta'], 1000.0000000003358, 9, "error bigger than 9 dp") 
        self.assertAlmostEqual(res.loc['AUDUSD Curncy 1mio 1SD USD delta'], -1642.9364799996902, 9, "error bigger than 9 dp") 
        self.assertAlmostEqual(res.loc['NZDUSD Curncy 1mio 1SD USD delta'], 2672.90927000019, 9, "error bigger than 9 dp") 
        self.assertAlmostEqual(res.loc['CADUSD Curncy 1mio 1SD USD delta'], -708.5169635927514, 9, "error bigger than 9 dp") 
        self.assertAlmostEqual(res.loc['CNHUSD Curncy 1mio 1SD USD delta'], 319.8132750057426, 9, "error bigger than 9 dp") 
        
        
class Test_coint_trading(unittest.TestCase):
    # johansen_test 
    def test_1(self):
        #res_jo = ct.johansen_test(df_fx_name_list_XXXUSD, name_list, False, 0, 1)
        
        res_estimation = ct.johansen_test_estimation(df_fx_name_list_XXXUSD, name_list, 0, 1)
        res_diag = ct.johansen_test_diag(res_estimation, df_fx_name_list_XXXUSD, name_list, False, 0, 1)

        self.assertAlmostEqual(res_diag['half life in days'], 17.860948621512236, 13, "error bigger than 13 dp")

        weights = res_diag['rounded weights'] 
        self.assertEqual(weights[0], 1.0) 
        self.assertEqual(weights[1], -1.6372584) 
        self.assertEqual(weights[2], 2.67379691) 
        self.assertEqual(weights[3], 1.29442965) 
        self.assertEqual(weights[4], -16.18005529)

        what_you_should_trade = res_diag['what you should trade'] 
        self.assertEqual(what_you_should_trade[0], 1.0) 
        self.assertEqual(what_you_should_trade[1], -1.6373) 
        self.assertEqual(what_you_should_trade[2], 2.6738) 
        self.assertEqual(what_you_should_trade[3], -0.9575) 
        self.assertEqual(what_you_should_trade[4], 2.2767)

        self.assertAlmostEqual(res_diag['what you should trade USD amount'], -0.5101475690219783, 13, "error bigger than 13 dp")
        self.assertAlmostEqual(res_diag['current zscore'], -0.45156084324827206, 13, "error bigger than 13 dp")

        self.assertTrue(res_estimation['trace (5%)'])
        self.assertTrue(res_estimation['eigen (5%)'])
        self.assertTrue(res_estimation['trace (10%)'])
        self.assertTrue(res_estimation['eigen (10%)'])

        self.assertAlmostEqual(res_diag['Johansen Basket'].iloc[-1], 0.5101475690219784, 13, "error bigger than 13 dp") 
        self.assertAlmostEqual(res_diag['Johansen Basket'].iloc[-2], 0.501817625816211, 13, "error bigger than 13 dp")

        self.assertAlmostEqual(res_diag['1mio 1SD in USD'], 50957.19656772441, 9, "error bigger than 9 dp")

        self.assertEqual(res_diag['PX_LAST'][0], 1.2551) 
        self.assertEqual(res_diag['PX_LAST'][1], 0.6897) 
        self.assertEqual(res_diag['PX_LAST'][2], 0.6371) 
        self.assertEqual(res_diag['PX_LAST'][3], 1.3519) 
        self.assertEqual(res_diag['PX_LAST'][4], 7.1068)

        self.assertAlmostEqual(res_diag['basket_std'], 0.050957196567724405, 13, "error bigger than 13 dp")
        self.assertAlmostEqual(res_diag['mean_rev_level'], 0.533157843673668, 13, "error bigger than 13 dp")

        self.assertAlmostEqual(res_diag['half life Sharpe Ratio'], 0.22248798311389345, 13, "error bigger than 13 dp")
        
        self.assertAlmostEqual(res_diag['RSI 14'], 65.31240575845882, 13, "error bigger than 13 dp")
        
    # extract_data_conditions_on_buy_sel1_signal 
    def test_2(self):
        lower_threshold = 2 
        higher_threshold = 5

        df_MeanRevStats = my_mean_rev.compute_sharpe_and_buysell_signal_multi_period(basket,
                                                                                     mean_rev_level,
                                                                                     basket_stdev,
                                                                                     mean_rev_rate_in_days, 
                                                                                     settings.HOLDING_PERIOD_IN_DAYS,
                                                                                     lower_threshold,
                                                                                     higher_threshold)

        df_buy_sell_signal = df_MeanRevStats[['Buy Signal', 'Sell Signal']]

        basket_x_days_hist_rtns = ut.compute_x_day_historical_returns(basket) 
        
        hist_rtns_times_1mio = basket_x_days_hist_rtns * settings.ONE_MILLION

        dates_to_buy, retns_to_buy, dates_to_sell, retns_to_sell = ct.extract_data_conditions_on_buy_sell_signal(hist_rtns_times_1mio,
                                                                                                                 df_buy_sell_signal,
                                                                                                                 TODAY) 

        self.assertEqual(dates_to_buy[0].strftime('%Y-%m-%d') , '2018-05-14') 
        self.assertEqual(dates_to_buy[-1].strftime('%Y-%m-%d'), '2020-05-18') 
        
        self.assertEqual(dates_to_sell[0].strftime('%Y-%m-%d') , '2015-10-15') 
        self.assertEqual(dates_to_sell[-1].strftime('%Y-%m-%d'), '2020-01-06')

        self.assertAlmostEqual(retns_to_buy[0], -2921.496300102744, delta=1e-8) 
        self.assertAlmostEqual(retns_to_buy[-1], 18964.68639085658, delta=1e-8)
        
        self.assertAlmostEqual(retns_to_sell[0], -4497.133724131408, delta=1e-8) 
        self.assertAlmostEqual(retns_to_sell[-1], -60391.157644035135, delta=1e-8)

    # This test is to mimick 'Search for cointegration basket' Jupter Notebook
    #
    # johansen_compute_all_baskets
    # filter_cointegration_basket_using_SD_threshold
    # filter_cointegration_basket_using_lm_1SD_in_USD 
    # filter_cointegration_basket_using_Sharpe_Ratio 
    # compute_johansen_stability_diagnostics
    # compute_historical_min_max_SD_diagnostics
    # compute_historical_return_diagnostics
    def test_3(self):
        start_list = ['2015-07-23', '2015-10-23'] 
        G8_no_CHF_list = ['EURUSD Curncy', 'GBPUSD Curncy', 'AUDUSD Curncy', 'NZDUSD Curncy', 
                          'JPYUSD Curncy', 'CADUSD Curncy', 'SGDUSD Curncy', 'CNHUSD Curncy']

        # -------------------------------------------------
        res_df = ct.johansen_compute_all_baskets(start_list, TODAY, G8_no_CHF_list, df_fx_spot)
        
        self.assertEqual(len(res_df), 420) 
        
        df_filtered = ct.filter_cointegration_basket_using_trace_10(res_df)
        
        self.assertEqual(len(df_filtered), 135)
        
        df_filtered = ct.compute_johansen_test_diag_for_all_coint_baskets(df_filtered, df_fx_spot)

        self.assertEqual(df_filtered['half life in days'].iloc[0], 44.996) 
        self.assertEqual(df_filtered['1mio 1SD in USD'].iloc[1], 42684.0)
        self.assertAlmostEqual(df_filtered['half life Sharpe Ratio'].iloc[2], 1.0076759436978293, 13, "error bigger than 13 dp")
        self.assertEqual(df_filtered['From'].iloc[3], '2015-07-23') 
        self.assertEqual(df_filtered['Today'].iloc[4], '2020-06-02')
        self.assertAlmostEqual(df_filtered['mean_rev_level'].iloc[5], 0.24878761311864384, 13, "error bigger than 13 dp")
        self.assertAlmostEqual(df_filtered['basket_std'].iloc[6], 0.042226190650347134, 13, "error bigger than 13 dp")
        
        self.assertEqual(df_filtered['eigen (5%)'].iloc[7], True) 
        self.assertEqual(df_filtered['eigen (10%)'].iloc[8], False) 
        self.assertEqual(df_filtered['trace (5%)'].iloc[9], False) 
        self.assertEqual(df_filtered['trace (10%)'].iloc[10], True)
        
        self.assertEqual(df_filtered['From'].iloc[134], '2015-10-23') 
        self.assertEqual(df_filtered['Today'].iloc[130], '2020-06-02')
        
        weights = df_filtered['unadj weights in XXXUSD'].iloc[11]
        self.assertEqual(weights[0], 1.0) 
        self.assertEqual(weights[1], -0.45514109)
        self.assertEqual(weights[2], 193.78412115)

        what_you_should_trade = df_filtered['what you should trade'].iloc[12] 
        self.assertEqual(what_you_should_trade[0], -1.0) 
        self.assertEqual(what_you_should_trade[1], -1.6596)
        self.assertEqual(what_you_should_trade[2], -3.1441)

        # ------------------------------
        SHOW_BASKET_WITH_SD_ABOVE = 1.3 
        SHOW_BASKET_WITH_Sharpe_ABOVE = 0.35
        SHOW_BASKET_WITH_HALF_LIFE_IN_DAYS_BELOW = 80

        # ------------------------------
        df_filtered = ct.filter_cointegration_basket_using_SD_threshold(df_filtered, SHOW_BASKET_WITH_SD_ABOVE) 
        self.assertEqual(df_filtered.index[0], 144) 
        self.assertEqual(df_filtered.index[-1], 92)

        # ------------------------------
        df_filtered = ct.filter_cointegration_basket_using_1m_1SD_in_USD(df_filtered) 
        self.assertEqual(df_filtered['1mio 1SD in USD'].iloc[0], 33104.0) 
        self.assertEqual(df_filtered['1mio 1SD in USD'].iloc[-1], 96673.0)

        # ------------------------------
        df_filtered = ct.filter_cointegration_basket_using_Sharpe_Ratio(df_filtered, SHOW_BASKET_WITH_Sharpe_ABOVE)
        self.assertAlmostEqual(df_filtered['half life Sharpe Ratio'].iloc[0], 1.5629416747185283, 13, "error bigger than 13 dp")
        self.assertAlmostEqual(df_filtered['half life Sharpe Ratio'].iloc[-1], 0.8047752894057187, 13, "error bigger than 13 dp")
        
        df_filtered = ct.filter_cointegration_basket_using_half_life_in_days(df_filtered, SHOW_BASKET_WITH_HALF_LIFE_IN_DAYS_BELOW)
        self.assertEqual(df_filtered['half life in days'].iloc[0], 16.604)
        self.assertEqual(df_filtered['half life in days'].iloc[-1], 27.255)

        # ------------------------------
        df_filtered = ct.compute_johansen_stability_diagnostics(df_filtered, df_fx_spot, settings.DataFreq.DAILY)
        self.assertEqual(df_filtered['+/- 1 month trace (10%)'].iloc[0], True) 
        self.assertEqual(df_filtered['+/- 1 month trace (10%)'].iloc[-1], True)
        self.assertEqual(df_filtered['Range in SD current'].iloc[0], 0.19) 
        self.assertEqual(df_filtered['Range in SD current'].iloc[-1], 0.17)

        # ------------------------------
        df_filtered = ct.compute_historical_min_max_SD_diagnostics(df_filtered, df_fx_spot)
        self.assertEqual(df_filtered['Stop Loss in SD'].iloc[0], 1.878)

        top_3_SD_Min = df_filtered['Top 3 SD Min'].iloc[-1]

        self.assertEqual(top_3_SD_Min[0], (-2.33, '2017-03-14')) 
        self.assertEqual(top_3_SD_Min[1], (-2.3,  '2017-03-16')) 
        self.assertEqual(top_3_SD_Min[2], (-2.28, '2017-03-13'))

        top_3_SD_Max = df_filtered['Top 3 SD Max'].iloc[-1]
        
        self.assertEqual(top_3_SD_Max[0], (3.24, '2015-10-30')) 
        self.assertEqual(top_3_SD_Max[1], (3.06, '2016-06-23')) 
        self.assertEqual(top_3_SD_Max[2], (2.97, '2016-06-21'))

        # ------------------------------
        # This is not used anymore but keep it here for future usage 
        #df_filtered = ct.computehhistorical_return_diagnostics(df_filtered, df_fxaspot)
        #self.assertEqual(df_filtered['hist 5D ave/std'].iloc[0], -0.63)

    # name_list_is_still_cointegrated
    def test_4(self):
        # we back test a basket that was traded on '2019-07-25' 
        local_FROM = '2014-04-23'
        local_TRADE_DATE = '2019-07-25'
        local_NOW = '2019-09-01'
        local_name_list = ['AUDUSD Curncy', 'JPYUSD Curncy', 'CADUSD Curncy', 'SGDUSD Curncy']
        local_df_fx_spot = df_fx_spot_all.loc[local_FROM:local_NOW]
        
        res_df = ct.name_list_is_still_cointegrated(local_FROM, local_TRADE_DATE, local_NOW, local_df_fx_spot, local_name_list)
        
        self.assertEqual(res_df['trace (10%)'].loc['2019-07-26'], True)
        self.assertEqual(res_df['trace (10%)'].loc['2019-07-31'], True)
        self.assertEqual(res_df['trace (10%)'].loc['2019-08-01'], False)


class Test_back_testing(unittest.TestCase):
    # name_list_string_to_name_list
    def test_1(self):
        name_str = "['EURUSD Curncy', 'GBPUSD Curncy', 'JPYUSD Curncy', 'SGDUSD Curncy', 'CNHUSD Curncy']" 
        local_name_list = btest.name_list_string_to_name_list(name_str)
        
        self.assertEqual(local_name_list[0], 'EURUSD Curncy') 
        self.assertEqual(local_name_list[1], 'GBPUSD Curncy') 
        self.assertEqual(local_name_list[2], 'JPYUSD Curncy') 
        self.assertEqual(local_name_list[3], 'SGDUSD Curncy') 
        self.assertEqual(local_name_list[4], 'CNHUSD Curncy')

    # weigths_list_string_to_float_list
    def test_2(self): 
        weights_XXXUSD_str = "[1.0, -1.64293648, 2.67290927, 1.29586726, -16.154956]" 
        local_weights_XXXUSD = btest.weigths_list_string_to_float_list(weights_XXXUSD_str)
        
        self.assertEqual(local_weights_XXXUSD[0], 1.0) 
        self.assertEqual(local_weights_XXXUSD[1], -1.64293648) 
        self.assertEqual(local_weights_XXXUSD[2], 2.67290927) 
        self.assertEqual(local_weights_XXXUSD[3], 1.29586726) 
        self.assertEqual(local_weights_XXXUSD[4], -16.154956)

    # back_test_one_trade 
    def test_3(self):
        # we back test a basket that was traded on '2020-01-15' with zscore at -1.508 at that time
        local_FROM = '2013-04-23'
        local_TRADE_DATE = '2020-01-15'
        local_NOW = TODAY
        local_name_list = ['AUDUSD Curncy', 'NZDUSD Curncy', 'CADUSD Curncy']
        local_weights_XXXUSD = [1.0, -0.77030425, -0.41245188] 
        zscore_TRADE_DATE = -1.508

        local_df_fx_spot = df_fx_spot_all.loc[local_FROM:local_NOW]

        res = btest.back_test_one_trade(local_FROM, local_TRADE_DATE, local_NOW, local_name_list, local_weights_XXXUSD, zscore_TRADE_DATE, local_df_fx_spot)

        self.assertAlmostEqual(res['5D Realized Rtns from Trade Date in SD'], -0.0939747144428266, delta=1e-11)
        self.assertAlmostEqual(res['5D max draw down in SD'], -0.19568917067110778, delta=1e-11) 
        self.assertAlmostEqual(res['basket stdev on Trade Date'], 0.017399315329440185, delta=1e-11)

    # This test is to mimick 'Search for cointegration basket Multi dates' Jupter Notebook 
    # back_test_many_trades
    def test_4(self):
        file = os.path.join(ROOT, 'unit_test_data/multi_dates_coint_search_for_back_testing_for_unit_test.xlsx')
        res_df_filtered = pd.read_excel(file)

        # we take the first 22 trades 
        res_df_filtered = res_df_filtered.iloc[:22]

        res_df_back_test = btest.back_test_many_trades(res_df_filtered, TODAY, df_fx_spot)

        self.assertEqual(res_df_back_test['FROM'].iloc[0], '2011-01-23') 
        self.assertEqual(res_df_back_test['Trade Date'].iloc[1], pd.Timestamp('2019-06-06 00:00:00'))
        self.assertEqual(res_df_back_test['Now'].iloc[2], '2020-06-02') 
        self.assertEqual(res_df_back_test['currency pairs'].iloc[3], ['AUDUSD Curncy', 'NZDUSD Curncy', 'JPYUSD Curncy', 'CADUSD Curncy']) 
        self.assertEqual(res_df_back_test['unadj weights in XXXUSD'].iloc[4], [1.0, -0.19479368, -20.80428566, -0.80133297])
        self.assertEqual(res_df_back_test['Sharpe Ratio on Trade Date'].iloc[5], 1.022758004808749)
        self.assertEqual(res_df_back_test['Z Score on Trade Date'].iloc[6], -1.879) 
        self.assertEqual(res_df_back_test['Abs Z Score on Trade Date'].iloc[7], 1.805) 
        self.assertEqual(res_df_back_test['Distance to max/min SD'].iloc[8], 0.639) 
        self.assertAlmostEqual(res_df_back_test['2D Realized Rtns in SD'].iloc[9], -0.08690412886190689, delta=1e-11) 
        self.assertAlmostEqual(res_df_back_test['5D Realized Rtns in SD'].iloc[10], -0.2162800513727469, delta=1e-11) 
        self.assertAlmostEqual(res_df_back_test['10D Realized Rtns in SD'].iloc[11], -0.6685227505117066, delta=1e-11) 
        self.assertAlmostEqual(res_df_back_test['2D max DD in SD'].iloc[12], -0.13902781042349244, delta=1e-11) 
        self.assertAlmostEqual(res_df_back_test['5D max DD in SD'].iloc[13], -0.27685180284259286, delta=1e-11) 
        self.assertAlmostEqual(res_df_back_test['10D max DD in SD'].iloc[14], -0.6995597233603178, delta=1e-11) 
        self.assertAlmostEqual(res_df_back_test['basket stdev on Trade Date'].iloc[15], 0.018752136531450955, delta=1e-11) 
        self.assertEqual(res_df_back_test['half life in days'].iloc[16], 40.237) 
        self.assertEqual(res_df_back_test['Range in SD current'].iloc[17], 0.07) 
        self.assertEqual(res_df_back_test['+/- 1 month trace (5%)'].iloc[18], True) 
        self.assertEqual(res_df_back_test['+/- 1 month trace (10%)'].iloc[19], True) 
        self.assertEqual(res_df_back_test['+/- 1 month eigen (5%)'].iloc[20], False) 
        self.assertEqual(res_df_back_test['+/- 1 month eigen (10%)'].iloc[21], True)


    # This test is to mimick 'Search for cointegration basket Multi dates' Jupter Notebook
    # back_test_many_trades 
    # one_back_test_summary_table
    def test_5(self) :
        file = os.path.join(ROOT, 'unit_test_data/multi_dates_coint_search_for_back_testing_for_unit_test.xlsx')
        res_df_filtered = pd.read_excel(file)

        # we take the first 50 trades 
        res_df_filtered = res_df_filtered.iloc[:50]

        res_df_back_test = btest.back_test_many_trades(res_df_filtered, TODAY, df_fx_spot)

        start_date = '2019-06-01' 
        end_date = '2020-01-01'

        Sharpe_threshold = 0.8 
        ZScore_threshold = 1.8

        res_df, my_trade_df = btest.one_back_test_summary_table(res_df_back_test, start_date, end_date, Sharpe_threshold, ZScore_threshold)

        self.assertEqual(res_df['Period Start'].iloc[0], '2019-06-01') 
        self.assertEqual(res_df['Period End'].iloc[1], '2020-01-01') 
        self.assertEqual(res_df['Sharpe threshold'].iloc[2], 0.8) 
        self.assertEqual(res_df['SD threshold'].iloc[2], 1.8) 
        self.assertEqual(res_df['Rtns type'].iloc[0], '2D ') 
        self.assertEqual(res_df['Pos %'].iloc[0], 0.344) 
        self.assertEqual(res_df['Pos %'].iloc[1], 0.031) 
        self.assertEqual(res_df['Pos %'].iloc[2], 0.0) 
        self.assertEqual(res_df['Neg %'].iloc[2], 1.0) 
        self.assertEqual(res_df['Total'].iloc[1], 32) 
        self.assertAlmostEqual(res_df['Rtns median'].iloc[0], -0.05824057438021002, delta=1e-11)
        self.assertAlmostEqual(res_df['Rtns median'].iloc[1], -0.1933344298555676, delta=1e-11) 
        self.assertAlmostEqual(res_df['Rtns median'].iloc[2], -0.5704432471100462, delta=1e-11) 
        self.assertAlmostEqual(res_df['Rtns mean'].iloc[0], -0.08422037167625586, delta=1e-11) 
        self.assertAlmostEqual(res_df['Rtns stdev'].iloc[0], 0.1339673984263569, delta=1e-11)

    # This test is to mimick 'Back Testing' Jupter Notebook 
    # several_back_test_summary_tables 
    def test_6(self):
        file = os.path.join(ROOT, 'unit_test_data/multi_dates_coint_search_for_back_testing_for_unit_test.xlsx')
        res_df_filtered = pd.read_excel(file)

        # we take the first 50 trades 
        res_df_filtered = res_df_filtered.iloc[:50]

        res_df_back_test = btest.back_test_many_trades(res_df_filtered, TODAY, df_fx_spot)

        start_dates = ['2019-06-01', '2019-06-07'] 
        end_dates   = ['2019-06-08', '2019-06-12']

        Sharpe_thresholds = [0.4, 0.8] 
        ZScore_thresholds = [1.5, 1.8]       

        df_back_test_summary_table = btest.several_back_test_summary_tables(res_df_back_test, 
                                                                            start_dates, 
                                                                            end_dates, 
                                                                            Sharpe_thresholds, 
                                                                            ZScore_thresholds)
        
        self.assertEqual(df_back_test_summary_table['Period Start'].iloc[0], '2019-06-01') 
        self.assertEqual(df_back_test_summary_table['Period End'].iloc[1], '2019-06-08') 
        self.assertEqual(df_back_test_summary_table['Sharpe threshold'].iloc[2], 0.4) 
        self.assertEqual(df_back_test_summary_table['SD threshold'].iloc[2], 1.5) 
        self.assertEqual(df_back_test_summary_table['Rtns type'].iloc[0], '2D ') 
        self.assertEqual(df_back_test_summary_table['Pos %'].iloc[0], 0.04) 
        self.assertEqual(df_back_test_summary_table['Pos %'].iloc[1], 0.04) 
        self.assertEqual(df_back_test_summary_table['Pos %'].iloc[2], 0.0) 
        self.assertEqual(df_back_test_summary_table['Neg %'].iloc[2], 1.0) 
        self.assertEqual(df_back_test_summary_table['Total'].iloc[1], 25) 
        self.assertAlmostEqual(df_back_test_summary_table['Rtns median'].iloc[0], -0.133054725586761, delta=1e-11) 
        self.assertAlmostEqual(df_back_test_summary_table['Rtns median'].iloc[1], -0.24195811142808563, delta=1e-11) 
        self.assertAlmostEqual(df_back_test_summary_table['Rtns median'].iloc[2], -0.5974346263520772, delta=1e-11) 
        self.assertAlmostEqual(df_back_test_summary_table['Rtns mean'].iloc[0], -0.1510282360577572, 13, "error bigger than 13 dp")
        self.assertAlmostEqual(df_back_test_summary_table['Rtns stdev'].iloc[0], 0.11931139202444883, delta=1e-11)

if __name__ == '__main__':
    # common data that is used for all test
    FROM = '2015-07-23'
    TODAY = '2020-06-02'
    name_list = ['GBPUSD Curncy', 'AUDUSD Curncy', 'NZDUSD Curncy', 'CADUSD Curncy', 'CNHUSD Curncy']
    weights_XXXUSD = [1.0, -1.64293648, 2.67290927, 1.29586726, -16.154956]

    df_fx_spot_all = myio.read_fx_daily_data('C:/temp/sdevpy/cointegration/unit_test_data/bloomberg fx data sheet_for_unit_test.xlsx') 
    df_fx_spot = df_fx_spot_all.loc[FROM:TODAY]
    df_fx_name_list_XXXUSD = df_fx_spot[name_list]

    basket = ut.compute_basket(df_fx_name_list_XXXUSD, weights_XXXUSD)

    mean_rev_ts = my_mean_rev.MeanRevTimeSeries(basket) 
    mean_rev_level = mean_rev_ts.get_mean_rev_level()
    basket_stdev = mean_rev_ts.get_stdev() 
    mean_rev_rate_in_days = mean_rev_ts.get_mean_rev_rate_in_days()

    unittest.main(warnings="ignore")
