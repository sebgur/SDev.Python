# import pandas as pd
# import os.path, time


def is_inverted_quote(key):
    is_inverted = False
    bbg_dict = {'EURUSD Curncy': False, 'GBPUSD Curncy': False, 'AUDUSD Curncy': False,
                'NZDUSD Curncy': False, 'JPYUSD Curncy': True, 'CADUSD Curncy': True,
                'CHFUSD Curncy': True, 'NOKUSD Curncy': True, 'SEKUSD Curncy': True,
                'SGDUSD Curncy': True, 'CNHUSD Curncy': True}

    is_inverted = bbg_dict.get(key)
    return is_inverted

