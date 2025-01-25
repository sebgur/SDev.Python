import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import utils as ut
import model_settings as settings
 
# -------------------------------------------
class MeanRevTimeSeries:

    # The init method or constructor  
    def __init__(self, time_series): 
        self.time_series = time_series
        
        # compute mean reversion statics
        res = compute_mean_reversion_params(self.time_series)
        
        self.half_life_in_days = res['Half Life in days']
        self.mean_rev_rate_in_days = res['Mean Rev Rate in days']
        self.mean_rev_level =  res['Mean Rev Level']
        
        # check if the OLS estimate is accurate or not. Smaller the value, the more accurate the result.
        self.const_pvalue = res['const p-value']
        self.Basket_pvalue = res['Basket p-value']   

        self.stdev = np.std(self.time_series)   
        
        # compute z score
        self.z_score_ts = (self.time_series - self.mean_rev_level)/self.stdev
        self.z_score_ts = self.z_score_ts.rename('z score')   

    def get_half_life_in_days(self):
        return self.half_life_in_days
 
    def get_mean_rev_rate_in_days(self):
        return self.mean_rev_rate_in_days
        
    def get_mean_rev_level(self):
        return self.mean_rev_level
        
    def get_const_pvalue(self):
        return self.const_pvalue    
    
    def get_Basket_pvalue(self):
        return self.Basket_pvalue
    
    # get the value at date 
    def get_level_at_t(self, date):
        return self.time_series.loc[date]
    
    # get the latest value in the time series
    def get_current_level(self):
        return self.time_series.iloc[-1]

    def get_stdev(self):
        return self.stdev
        
    def get_zscores_time_series(self):
        return self.z_score_ts

    def get_current_zscore(self):
        return self.z_score_ts.iloc[-1]        
        
    # End of the class MeanRevTimeSeries       
        
        
        
# -------------------------------------------
# dX(t) = lam * (mu - X(t))*dt + BM
# mean_rev_level - mu
# mean_rev_rate_in_days - mean rev rate (assume is -ve) in days
# time in days - because we estimate using daily data
# current_level - current level of basket, i.e. X(0)
# daily_hist_normal_vol - daily standard dev of the basket 
def mean_rev_expected_and_variance_change(mean_rev_level, mean_rev_rate_in_days, time_in_days, current_level, daily_hist_normal_vol):
    
    exp_lam_T = np.exp(mean_rev_rate_in_days * time_in_days)
    exp_2lam_T = np.exp(2.0 * mean_rev_rate_in_days * time_in_days)   
    
    level_at_T = current_level * exp_lam_T + mean_rev_level * (1.0 - exp_lam_T)      

    # expectation of return in time_in_days
    EdX = level_at_T - current_level        
    
    daily_var = daily_hist_normal_vol * daily_hist_normal_vol
    
    # variance of return in time_in_days    
    vardX = daily_var/(-2.0*mean_rev_rate_in_days)*(1.0 - exp_2lam_T)  
    
    return EdX, vardX 

# -------------------------------------------    
def compute_sharpe_ratio(mean_rev_level, mean_rev_rate_in_days, time_in_days, current_level, daily_hist_normal_vol, current_zscore):   
     
    mean_S, var_S = mean_rev_expected_and_variance_change(mean_rev_level, 
                                                          mean_rev_rate_in_days, 
                                                          time_in_days, 
                                                          current_level, 
                                                          daily_hist_normal_vol)    
    return_expectation_over_T = 0    
    return_vol_over_T = np.sqrt(var_S)    
    
    if current_zscore < 0:     
        # We buy the basket
        return_expectation_over_T = mean_S        
    else:
        # We short the basket
        return_expectation_over_T = -mean_S 

    sharpe_ratio = return_expectation_over_T/return_vol_over_T      

    res_dict = {'Sharpe Ratio': sharpe_ratio,
                'Return Expectation': return_expectation_over_T,
                'Return SD': return_vol_over_T
               }
        
    return res_dict
    
#------------------------------------------------------------------------------
# we buy/sell if the current zscore is -ve/+ve
# only if the current zscore is between abs_lower_threshold and abs_higher_threshold
# with respect to the correct signal
def compute_buy_sell_signal(current_zscore, abs_lower_threshold, abs_higher_threshold):

    assert abs_lower_threshold < abs_higher_threshold, 'abs_lower_threshold >= abs_higher_threshold'
    
    #--------------------------------------------------------
    # 0 means no action , 1 means either buy or sell
    buy_signal = 0
    sell_signal = 0
    
    if -abs_higher_threshold < current_zscore and current_zscore < -abs_lower_threshold:
        buy_signal = 1
    elif abs_lower_threshold < current_zscore and current_zscore < abs_higher_threshold:
        sell_signal = 1
        
    return buy_signal, sell_signal
    
# -----------------------------------------------------------------------    
# We give a fixed basket, mean rev level, basket_std and mean_rev_rate_in_days
# then we compute the sharpe ratio
# We also compute the buy sell signal if the current threshold is within the thresholds
# Note that the thresholds are always positive numbers.
def compute_sharpe_and_buysell_signal_multi_period(basket,
                                                   mean_rev_level,
                                                   basket_stdev,
                                                   mean_rev_rate_in_days,
                                                   holding_period_in_days,
                                                   lower_threshold,
                                                   higher_threshold):
                                                   
    # compute daily historical basket normal vol time series. Default num of period = 15
    # in other words, normal vol times series is 15 periods shorter than the basket
    normal_vol_ts = ut.compute_daily_hist_normal_vol(basket)
    zscores_ts = zscores_mean_revert_time_series(basket, mean_rev_level, basket_stdev)
    
    output = []
    
    # we loop throught the whole history of the time series
    for t in normal_vol_ts.index:
        zscore_at_t =  zscores_ts.loc[t]
        
        res_sharpe = compute_sharpe_ratio(mean_rev_level,
                                          mean_rev_rate_in_days,
                                          holding_period_in_days,
                                          basket.loc[t],
                                          normal_vol_ts.loc[t],
                                          zscore_at_t)
                                          
        buy_signal, sell_signal = compute_buy_sell_signal(zscore_at_t, lower_threshold, higher_threshold)
        
        output.append((t,
                       res_sharpe['Sharpe Ratio'],
                       res_sharpe['Return Expectation'],
                       res_sharpe['Return SD'],                       
                       buy_signal,
                       sell_signal
                      ))
                      
        # end of for t in normal_vol_ts.index
    
    df_output = pd.DataFrame(output, columns =['Dates',
                                               'Sharpe Ratio', 
                                               'Return Expectation over X days', 
                                               'Return SD over X days',
                                               'Buy Signal',
                                               'Sell Signal'
                                               ])
    
    df_output = df_output.set_index('Dates')

    return df_output
    
         
# ---------------------------------------------------------------------------         
def zscores_mean_revert_time_series(basket, mean_rev_level, stdev):   
    z_score_ts = (basket - mean_rev_level)/stdev
    z_score_ts = z_score_ts.rename('z score')         
    return z_score_ts
    
    
# ---------------------------------------------------------------------------
# compute mean reversio parameters:
# (1) half life
# (2) mean reversion level
# (3) mean reversion rate
# (4) p values of the OLS estimation. The smaller the p-value, the better.
def compute_mean_reversion_params(s): 

    # compute the diff and the shift the position by -1 so that we have dS(t) vs S(t-1)
    ds = s.diff().shift(-1)     
    
    # skip the last element which is NA    
    ds = ds.iloc[:-1]           
    
    # skip the last element    
    s = s.iloc[:-1]                   

    # perform regression: dS(t) = a + b * S(t-1)    
    s_const = sm.add_constant(s)    
    results = sm.OLS(ds, s_const).fit() 
    
    # if we assume dS(t) = lambda (S_bar - S(t-1))dt + \sigma dW(t)
    # then
    # a = lambda * S_bar * dt
    # b = -lambda * dt
        
    a = results.params['const']

    b = results.params['Basket'] 
    
    # see clewlow and strickland's energy derivatives pricing and risk management p28, 29
    # this is the proper way to do it, not using np.mean(basket) to compute the mean
    mean_rev_level = -a/b
    
    # we expect this is a positive number. This is just a convention that quantopian use and I follow that.
    if b > 0:
        print('The series is not mean reverting')
    
    # solution of the equation: 1/2 = exp(b*T) -> T = -ln(2)/b
    half_life_in_days = -np.log(2) / b   
    
    # this is -lambda * dt, where dt depends on the data freq. If daily, dt = 1/365.
    # To use this later, all we need is to put the number of days rather than year fraction
    # and we don't need to put the minus sign
    # e.g. 5 days -> exp(mean_rev_rate_in_days * 5) NOT exp(mean_rev_rate_in_days * 5/365)
    mean_rev_rate_in_days = b
    
    res_dict = {'Half Life in days': half_life_in_days,
                'Mean Rev Rate in days': mean_rev_rate_in_days,
                'Mean Rev Level': mean_rev_level,
                'const p-value': results.pvalues['const'], 
                'Basket p-value': results.pvalues['Basket'],
                }
    
    return res_dict
        



# df_fx_XXXUSD - dataframe contains fx spot rate data in XXXUSD, index = date
# name_list -  a list of currency basket
# weights_XXXUSD - cointegrated weights of the basket in XXXUSD
def compute_zscore_for_a_fixed_basket_for_multi_period(df_fx_XXXUSD, name_list, weights_XXXUSD):

    df_fx_name_list_XXXUSD = df_fx_XXXUSD[name_list]  

    # Note that the weights are given here. This means that we are NOT estimating the weights are each day.
    # If we were to estimate (i.e. Running Johansen) then we will have DIFFERENT weights for each day.
    # This means this is NOT equivalent to running the estimation everyday using the latest data.
    # This is checking the historical performace of a particular basket with a fixed weightings.
    basket = ut.compute_basket(df_fx_name_list_XXXUSD, weights_XXXUSD)   
    
    mean_rev_ts = MeanRevTimeSeries(basket)

    z_score_ts = zscores_mean_revert_time_series(basket, mean_rev_ts.get_mean_rev_level(), mean_rev_ts.get_stdev())
    
    return z_score_ts

