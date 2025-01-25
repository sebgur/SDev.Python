


import pandas as pd   
import os.path, time    

#--------------------------------------------------------------------------
# for excel output style
def color_negative_red(value_numeric):    
    if value_numeric < 0:
        color = 'red'
    elif value_numeric > 0:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color

#--------------------------------------------------------------------------  
# for excel output style  
def color_TRUE_FALSE(value_bool):
    
    if value_bool:
        color = 'green'
    else:
        color = 'red'

    return 'background-color: %s' % color

#--------------------------------------------------------------------------    
def format_cointegration_seach_output(df):

    df_formatted = df.style.applymap(color_TRUE_FALSE, subset=['trace (5%)', 
                                                               'trace (10%)',
                                                               'eigen (5%)', 
                                                               'eigen (10%)',
                                                               '+/- 1 month trace (5%)',
                                                               '+/- 1 month trace (10%)',
                                                               '+/- 1 month eigen (5%)',
                                                               '+/- 1 month eigen (10%)'
                                                               ])
                                                               
    return df_formatted
        
#--------------------------------------------------------------------------
def format_rolling_dates_output(df):

    df_formatted = df.style.applymap(color_TRUE_FALSE, subset=['trace (5%)', 
                                                               'trace (10%)',
                                                               'eigen (5%)', 
                                                               'eigen (10%)'
                                                               ])
                                                               
    return df_formatted

#--------------------------------------------------------------------------
def read_all_fx_data_name_list_and_compute_ivol_MA(xls_file_name, FROM, TODAY, name_list):

    # read in the data from bloomberg
    df_fx_XXXUSD, df_fx_ivol = read_all_fx_data(xls_file_name)

    # slicing the data for fx spot
    df_fx_XXXUSD = df_fx_XXXUSD.loc[FROM:TODAY]
    df_fx_name_list_XXXUSD = df_fx_XXXUSD[name_list]

    # slicing the data for fx ivol
    df_fx_ivol = df_fx_ivol.loc[FROM:TODAY]
    df_fx_ivol_name_list = df_fx_ivol[name_list]

    # compute fx currency pairs ivol with moving average
    df_ivol_with_MA_name_list = compute_fx_ivol_with_MA(df_fx_ivol, name_list)
    
    res_dict = {'df_fx_name_list_XXXUSD': df_fx_name_list_XXXUSD,
                'df_fx_ivol_name_list': df_fx_ivol_name_list,
                'df_ivol_with_MA_name_list': df_ivol_with_MA_name_list
                }
                
    return res_dict



# ---------------------------------------------------------------------------  
def is_inverted_quote(key):    
    is_inverted = False

    bbg_dict = {'EURUSD Curncy': False, 
                'GBPUSD Curncy': False, 
                'AUDUSD Curncy': False, 
                'NZDUSD Curncy': False,             
                'JPYUSD Curncy': True, 
                'CADUSD Curncy': True, 
                'CHFUSD Curncy': True, 
                'NOKUSD Curncy': True, 
                'SEKUSD Curncy': True,    
                'SGDUSD Curncy': True, 
                'CNHUSD Curncy': True      
                }
            
    is_inverted = bbg_dict.get(key)   
    
    return is_inverted
   
#---------------------------------------------------------------------------   
# XXXUSD Curncy to USDXXX Curncy or vice-versa
def convert_name_ordering(name):    
    new_name = name[3:6] + name[0:3] + str(' Curncy')    
    return new_name

 
#---------------------------------------------------------------------------
def convert_list_to_market_conv(name_list):

    N = len(name_list)    
    res_list = []
    
    for i in range(N):       
        name = name_list[i]       
        
        if is_inverted_quote(name) == True:
            new_name = convert_name_ordering(name)
            res_list.append(new_name)
        else:
            res_list.append(name)

    return res_list


# ---------------------------------------------------------------------------    
def compute_fx_ivol_with_MA(df_fx_ivol, name_list): 
    
    df_fx_ivol_name_list = df_fx_ivol[name_list]
    
    df_ivol_ma = df_fx_ivol.rolling(120).mean()
	
    df_ivol_ma_name_list = df_ivol_ma[name_list]

    # rename all the column names to XXXUSD for easier retrival later
    # note that ATM XXXUSD and ATM USDXXX vol is the same
    df_ivol_ma_name_list = df_ivol_ma_name_list.rename(columns={'EURUSD Curncy': 'EURUSD MA',
                                                                'GBPUSD Curncy': 'GBPUSD MA', 
                                                                'AUDUSD Curncy': 'AUDUSD MA', 
                                                                'NZDUSD Curncy': 'NZDUSD MA',      
                                                                'JPYUSD Curncy': 'JPYUSD MA',    
                                                                'CADUSD Curncy': 'CADUSD MA',
                                                                'CHFUSD Curncy': 'CHFUSD MA', 
                                                                'NOKUSD Curncy': 'NOKUSD MA', 
                                                                'SEKUSD Curncy': 'SEKUSD MA',     
                                                                'SGDUSD Curncy': 'SGDUSD MA',
                                                                'CNHUSD Curncy': 'CNHUSD MA'                        
                                                                })
    
    df_ivol_with_MA_name_list = pd.concat([df_fx_ivol_name_list, df_ivol_ma_name_list], axis=1)
    
    return df_ivol_with_MA_name_list
    
# ---------------------------------------------------------------------------    
def read_all_fx_data(xls_file_name):

    xls = pd.ExcelFile(xls_file_name)

    df_fx_spot = xls.parse('spot', index_col='Dates')     
    df_fx_spot = df_spot_fx_to_XXXUSD(df_fx_spot)    
    
    from_date = df_fx_spot.iloc[1].name.strftime('%Y-%m-%d')    
    to_date = df_fx_spot.iloc[-1].name.strftime('%Y-%m-%d')
    
    print('FX spot data set is from = ' + str(from_date) + ' to ' + str(to_date))
    
    df_fx_ivol = xls.parse('ivol', index_col='Dates') 
    
    # rename all the column names to XXXUSD for easier retrival later
    # note that the ATM implied vol for XXXUSD vs USDXXX is the same.
    df_fx_ivol = df_fx_ivol.rename(columns={'EURUSDV1M Curncy': 'EURUSD Curncy',
                                            'GBPUSDV1M Curncy': 'GBPUSD Curncy', 
                                            'AUDUSDV1M Curncy': 'AUDUSD Curncy', 
                                            'NZDUSDV1M Curncy': 'NZDUSD Curncy',      
                                            'USDJPYV1M Curncy': 'JPYUSD Curncy',    
                                            'USDCADV1M Curncy': 'CADUSD Curncy',
                                            'USDCHFV1M Curncy': 'CHFUSD Curncy', 
                                            'USDNOKV1M Curncy': 'NOKUSD Curncy', 
                                            'USDSEKV1M Curncy': 'SEKUSD Curncy',     
                                            'USDSGDV1M Curncy': 'SGDUSD Curncy',
                                            'USDCNHV1M Curncy': 'CNHUSD Curncy'                        
                                            })
    
    from_date = df_fx_ivol.iloc[1].name.strftime('%Y-%m-%d')    
    to_date = df_fx_ivol.iloc[-1].name.strftime('%Y-%m-%d')
    
    print('FX ivol data set is from = ' + str(from_date) + ' to ' + str(to_date))
    
    print('Data is up to date at : %s' %time.ctime(os.path.getmtime(xls_file_name)))
    
    return df_fx_spot, df_fx_ivol

# ---------------------------------------------------------------------------  
def read_fx_daily_data(xls_file_name):

    xls = pd.ExcelFile(xls_file_name)

    df_fx_spot = xls.parse('spot', index_col='Dates')     
    df_fx_spot = df_spot_fx_to_XXXUSD(df_fx_spot)    
    
    from_date = df_fx_spot.iloc[1].name.strftime('%Y-%m-%d')    
    to_date = df_fx_spot.iloc[-1].name.strftime('%Y-%m-%d')
    
    print('FX spot data set is from = ' + str(from_date) + ' to ' + str(to_date))
    
    print('Data is up to date at : %s' %time.ctime(os.path.getmtime(xls_file_name)))
    
    return df_fx_spot

   
# ---------------------------------------------------------------------------  
def df_spot_fx_to_XXXUSD(df_fx):

    df_fx['USDJPY Curncy'] = 1 / df_fx['USDJPY Curncy']
    df_fx['USDCAD Curncy'] = 1 / df_fx['USDCAD Curncy']
    df_fx['USDCHF Curncy'] = 1 / df_fx['USDCHF Curncy']
    df_fx['USDNOK Curncy'] = 1 / df_fx['USDNOK Curncy']
    df_fx['USDSEK Curncy'] = 1 / df_fx['USDSEK Curncy']

    df_fx['USDSGD Curncy'] = 1 / df_fx['USDSGD Curncy']
    df_fx['USDCNH Curncy'] = 1 / df_fx['USDCNH Curncy']

    # rename the column names
    df_fx = df_fx.rename(columns={'USDJPY Curncy': 'JPYUSD Curncy', 
                                  'USDCAD Curncy': 'CADUSD Curncy',
                                  'USDCHF Curncy': 'CHFUSD Curncy', 
                                  'USDNOK Curncy': 'NOKUSD Curncy', 
                                  'USDSEK Curncy': 'SEKUSD Curncy',     
                                  'USDSGD Curncy': 'SGDUSD Curncy',
                                  'USDCNH Curncy': 'CNHUSD Curncy'
                                  })


    return df_fx    

