# Import packages
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from sdevpy.tools import utils
from openbb import obb
obb.user.preferences.output_type = "dataframe"

names = ['SPY', 'VTV', 'VUG', 'VBR', 'VBK', 'VGK', 'VPL']
colors = ['blue', 'red', 'green', 'brown', 'orange', 'yellow']
today = dt.date.today()
hist_window = utils.DateTimeSpan(0, 0, 1)  # days, months, years
db_root = r'C:\\temp\\database'

print(today.strftime('%d-%b-%Y'))

# Retrieving all data
hist_start = utils.date_advance(today, hist_window)
print(hist_start.strftime('%d-%b-%Y'))

start = "2025-03-01"
end = "2025-03-16"
df = obb.equity.price.historical(names[0], provider="yfinance", start_date=dt.date(2025, 5, 1),
                                     end_date=dt.date(2025, 5, 25))
# df = df['close']
# df.drop(columns=['volume', 'dividend'], inplace=True)
print(df.head())