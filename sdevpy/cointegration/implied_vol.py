import numpy as np
import pandas as pd
from scipy.stats import norm
import utils as ut
import data_io as myio
 
# Compute historical corelation with time series in XXX_USD
# The output corr matrix is in the format of market convention quote
def historical_corr_matrix(time_series_in_XXXUSD):

    # compute the correlation of the return of the time series in XXX_USD
    res_df = time_series_in_XXXUSD.pct_change().corr()
    
    name_list = res_df.keys()

    for ccy1 in name_list:
        bool_is_ccy1_inv = myio.is_inverted_quote(ccy1)

        for ccy2 in name_list:
            bool_is_ccy2_inv = myio.is_inverted_quote(ccy2)                
            
            b_2  = bool_is_ccy1_inv != bool_is_ccy2_inv
            
            # if only one of the quotes is not in market convention, then we change the sign of the correlation
            if bool_is_ccy1_inv != bool_is_ccy2_inv:         
                res_df.loc[[ccy1], [ccy2]] = -1 * res_df.loc[[ccy1], [ccy2]]    
    
    
    return res_df
    
    
# return an array fill with N elements of 1/N
def create_equal_weights_array(N):
    res = np.zeros(N)                    
    res.fill(1/N)
    return res
    
# take in an array of weights which can be +/-.
# sum the abs of all elements and then normalized using the sum of abs
def create_normalized_weights_array(notionals):
    sum_abs_notionals = (np.abs(notionals)).sum()   
    res = notionals / sum_abs_notionals     

    return res
   
# --------------------------------------------------------------- 
# fx_spot_data - in the convention of XXXUSD
# fx_ivol_data - fx implied vol market data 
# weights - co-integrating factors in the unit of XXXUSD
def implied_lognormal_vol_fx_basket(fx_spot_data_in_XXXUSD, fx_ivol_data, weights):
    
    # in the convention of XXXUSD
    name_list_in_XXXUSD = fx_spot_data_in_XXXUSD.keys()
  
    # corr matrix in market convention quote
    corr_matrix = historical_corr_matrix(fx_spot_data_in_XXXUSD)
   
    #ivol for each pairs
    latest_ivol = fx_ivol_data[name_list_in_XXXUSD].iloc[-1]
    
    # make it to be diagonal matrix
    v = np.diag(latest_ivol.values)
    
    # compute covariance matrix using implied vol and historical correlation
    covar = np.matmul(np.matmul(v, corr_matrix), v)        
           
    # ---- assume equally weighting for simplicity   
    normalized_notionals = create_equal_weights_array(len(weights))    
    
    ivol_fx_basket = np.sqrt(np.dot(np.dot(normalized_notionals, covar), normalized_notionals))    
    
    res_dict = {'Basket lognormal ivol': ivol_fx_basket,
                'Normalized Notionals': normalized_notionals,
                'Correlation Matrix': corr_matrix,
                'FX lognormal ivol': latest_ivol
                }
                
    return res_dict

# --------------------------------------------------------------- 
# This function computes the fx basket implied volatility using historical corre and current ivol data
# fx_spot_data - in the convention of XXXUSD
# fx_ivol_data - fx implied vol market data 
# weights - co-integrating factors in the unit of XXXUSD
def compute_fx_basket_ivols_and_ma(start_date, end_date, fx_spot_data_in_XXXUSD, fx_ivol_data, weights_XXXUSD):

    basket_ivol = []

    sd = pd.to_datetime(start_date, format='%Y-%m-%d') 
    ed = pd.to_datetime(end_date, format='%Y-%m-%d') 

    for t in fx_spot_data_in_XXXUSD.index:
        
        if t <= ed and t > sd:
        
            data_fx_spot_at_t = fx_spot_data_in_XXXUSD.loc[:t]
            data_fx_ivol_at_t = fx_ivol_data.loc[:t] 

            res_dict_vol = implied_lognormal_vol_fx_basket(data_fx_spot_at_t, 
                                                           data_fx_ivol_at_t, 
                                                           weights_XXXUSD)

            basket_ivol.append((t, res_dict_vol['Basket lognormal ivol'] ) )    
        
    df_basket_ivol = pd.DataFrame(basket_ivol, columns=['Dates', 'Basket ivol'])            
    df_basket_ivol = df_basket_ivol.set_index('Dates')    
    
    df_basket_ivol_ma = df_basket_ivol.rolling(120).mean()    
    df_basket_ivol_ma = df_basket_ivol_ma.rename(columns={'Basket ivol': 'Basket ivol 120 period MA'})
    
    df_basket_ivol_with_ma = pd.concat([df_basket_ivol, df_basket_ivol_ma], axis=1)
    
    return df_basket_ivol_with_ma


    