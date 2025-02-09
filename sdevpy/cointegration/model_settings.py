from enum import Enum
import pandas as pd

class ToShiftOrNot(Enum):
    TO_SHIFT = 1
    NOT_SHIFT = 0

ONE_MILLION = 1000000

# number of days which we hold the trade for
HOLDING_PERIOD_IN_DAYS = 5

# number of period to be used for historical vol estimation
NUM_PERIOD_FOR_HIST_VOL_EST = 15

# define the range of buying and selling for a given SD. E.g. if the target 1.5, SAUSAGE_THICKNESS = 0.25, we buy/sell if 1.25 < SD < 1.75
SAUSAGE_THICKNESS = 0.25

# the date that Bloomberg has USDCNH fx spot price
CNH_BEGINNING_DATE = pd.to_datetime('2010-08-23', format='%Y-%m-%d')

# the Swiss National Bank choose not to defend EURCHF at 1.2
SNB_DATE = pd.to_datetime('2015-01-15', format='%Y-%m-%d')

# To determine we are varying the start date or end date of the data when performing Johansen stability test 
class JohansenVaringDate(Enum):
    START = 0 # The range is for the start date of the dataset 
    END = 1 # The range is for the end date of the dataset 

# Are we dealing with Daily data or intraday data?
class DataFreq(Enum):
    INTRADAY = 0
    DAILY = 1