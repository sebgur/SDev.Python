import os
import datetime as dt
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sdevpy.cointegration import data_io as myio


def johansen_estimation(df_data, det_order = 0, k_ar_diff = 1):
    """ Estimate the weights and test statistics """
    # Run Johansen test 
    res_jo = coint_johansen(df_data, det_order, k_ar_diff)

    # Check the trace and eigenvalue test of Johansen 
    trace_5pct, trace_10pct, eigen_5pct, eigen_10pct = check_johansen_stats_fast(res_jo)

    # Get the normalized lst eigenvector which is the weights 
    weights = norm_1st_eigvec(res_jo)

    return {'weights': weights, 'trace (5%)': trace_5pct, 'eigen (5%)': eigen_5pct, 
            'trace (10%)': trace_10pct, 'eigen (10%)': eigen_10pct}


def check_johansen_stats_fast(res_jo):
    """ Extract test stats for trace/eigen at 5% and 10% confident intervals """
    
    # Trace_stats
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
            
   # Eigen stats           
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


def norm_1st_eigvec(res_jo):
    """ Retrieve the normalized first eigen vector from the result of Johansen test.
        The normalized vector are the weights of the cointegrated basket """
    # Size of the eigenvector
    N = len(res_jo.evec)    
    first_eigvec = []
    for i in range(N):
        first_eigvec.append(res_jo.evec[i][0])
    
    # Normalized with respect to the first element
    return np.array(first_eigvec) / res_jo.evec[0][0]


def convert_to_currency(df, target_ccy):
    """ Convert FX spot data to target currency """
    converted_df = df
    converted_df.set_index('Dates')
    for col in converted_df.columns[1:]:
        is_usd_for, inverse_ticker = is_fx_for_ticker(col, target_ccy)
        if is_usd_for:
            converted_df[col] = 1.0 / converted_df[col]
            converted_df = converted_df.rename(columns={col: inverse_ticker})

    return converted_df


def is_fx_for_ticker(ticker, for_ccy='USD'):
    """ If ticker is for FX spot and for_ccy is the foreign currency, answer True and
        the inverted ticker. Otherwise answer false and empty string """
    splits = str.split(ticker, " ")
    pair = splits[0]
    is_fx_for = False
    inverse_ticker = ""
    if len(pair) == 6 and pair[0:3].upper() == for_ccy:
        is_fx_for = True
        inverse_ticker = pair[3:6].upper() + pair[0:3].upper() + ticker[6:]

    return is_fx_for, inverse_ticker


if __name__ == "__main__":
    ROOT = r"C:\\temp\\sdevpy\\cointegration"
    FROM = '2015-07-23'
    TODAY = '2020-06-02'
    FROM_DATE = dt.date(2015, 7, 23)
    TO_DATE = dt.date(2020, 6, 2)
    ticker_list = ['GBPUSD Curncy', 'AUDUSD Curncy', 'NZDUSD Curncy', 'CADUSD Curncy', 'CNHUSD Curncy']

    # # Convert FX spots into USD and output converted data
    # data_file = os.path.join(ROOT, "test.tsv")
    # df_data = pd.read_csv(data_file, sep='\t')
    # df_data.set_index('Dates')
    # converted_df = convert_to_currency(df_data, "USD")
    # output_file = os.path.join(ROOT, "converted.tsv")
    # converted_df.to_csv(output_file, index=False, sep='\t')

    # New data
    print("<><><><> Running NEW <><><><>")
    data_file = os.path.join(ROOT, "fx_spots.tsv")
    df_data = pd.read_csv(data_file, sep='\t')
    dates_str = df_data['Dates']
    dates = [dt.datetime.strptime(x, "%Y-%m-%d").date() for x in dates_str]
    df_data['Dates'] = dates
    df_fx_spots = df_data[df_data['Dates'] >= FROM_DATE]
    df_fx_spots = df_fx_spots[df_fx_spots['Dates'] <= TO_DATE]
    # print(df_data.to_string(max_rows=6, max_cols=6))

    print(df_fx_spots.head())
    df_data = df_fx_spots[ticker_list]
    print(df_data.head())
    estimation = johansen_estimation(df_data, 0, 1)
    print(estimation)

    # Old data
    print("<><><><> Running OLD <><><><>")
    data_file_xls = os.path.join(ROOT, "unit_test_data/bloomberg fx data sheet_for_unit_test.xlsx")
    df_data_xls = myio.read_fx_daily_data(data_file_xls)
    df_fx_spots_xls = df_data_xls.loc[FROM:TODAY]
    df_fx_spots_xls = df_fx_spots_xls[ticker_list]

    print(df_fx_spots_xls.head())
    df_data_xls = df_fx_spots_xls[ticker_list]
    print(df_data_xls.head())
    estimation_xls = johansen_estimation(df_data_xls, 0, 1)
    print(estimation_xls)
