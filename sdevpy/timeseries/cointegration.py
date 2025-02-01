import os
import datetime as dt
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sdevpy.cointegration import data_io as myio


def johansen_test_estimation(data, asset_name_list, det_order = 0, k_ar_diff = 1):
    """ Estimate the weights and test statistics """
    # get the data according to the asset name list 
    df_name_list = data[asset_name_list]

    # Run Johansen test 
    res_jo = coint_johansen(df_name_list, det_order, k_ar_diff)

    # check the trace and eigenvalue test of Johansen 
    # bool_trace_5pct, bool_trace_10pct, bool_eigen_5pct, bool_eigen_10pct = check_johansen_test_stats_fast(res_jo)

    # get the normalized lst eigenvector which is the weights 
    # weights_XXXUSD = norm_1st_eigvec(res_jo)

    # res_dict = {'weights': weights_XXXUSD, 
    #             'trace (5%)': bool_trace_5pct, 
    #             'eigen (5%)': bool_eigen_5pct, 
    #             'trace (10%)': bool_trace_10pct, 
    #             'eigen (10%)': bool_eigen_10pct
    #             }

    return res_jo


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

    data_file = os.path.join(ROOT, "fx_spots.tsv")
    df_data = pd.read_csv(data_file, sep='\t')
    dates_str = df_data['Dates']
    dates = [dt.datetime.strptime(x, "%Y-%m-%d").date() for x in dates_str]
    # print(df_data.to_string(max_rows=4, max_cols=6))
    df_data['Dates'] = dates
    # print(df_data.to_string(max_rows=4, max_cols=6))

    df_fx_spots = df_data[df_data['Dates'] >= FROM_DATE]
    df_fx_spots = df_fx_spots[df_fx_spots['Dates'] <= TO_DATE]
    print(len(df_fx_spots))
    # print(df_fx_spots.to_string(max_rows=6, max_cols=6))

    # df_fx_spots = df_data.loc[FROM:TODAY]
    # print(df_fx_spots.to_string(max_rows=6, max_cols=6))
    # df_fx_spots.set_index('Dates')
    # df_fx_spots = df_fx_spots[ticker_list]
    # print(df_fx_spots.to_string(max_rows=8, max_cols=6))
    # print(len(df_fx_spots))

    data_file_xls = os.path.join(ROOT, "unit_test_data/bloomberg fx data sheet_for_unit_test.xlsx")
    df_data_xls = myio.read_fx_daily_data(data_file_xls)
    # print(df_data_xls.dtypes)
    # print(df_data_xls['Dates'].dtype)
    print(df_data_xls.to_string(max_rows=4, max_cols=6))

    df_fx_spots_xls = df_data_xls.loc[FROM:TODAY]
    print(len(df_fx_spots_xls))
    df_fx_spots_xls = df_fx_spots_xls[ticker_list]
    # print(df_fx_spots_xls.to_string(max_rows=4, max_cols=6))

    # print("Running NEW")
    # # print(df_fx_spots.head())
    # # estimation = johansen_test_estimation(df_fx_spots, ticker_list, 0, 1)

    # print("Running OLD")
    # # estimation_xls = johansen_test_estimation(df_fx_spots_xls, ticker_list, 0, 1)

