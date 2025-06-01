# Import packages
import os
import datetime as dt
import pandas as pd
from dateutil.relativedelta import relativedelta
from sdevpy.tools import utils
from sdevpy.tools import filemanager
from openbb import obb
obb.user.preferences.output_type = "dataframe"

db_path = r'C:\\temp\\database'
end = dt.date.today()


# Collect names
ext = '.tsv'
all_files = filemanager.list_files(db_path, extensions=[ext])
names = [f.replace(ext, "") for f in all_files]

print(f"Found names: {len(names)}")
print(names)

# for name in names:
name ='^SPX'
df = obb.equity.price.historical(name, provider="yfinance", start_date=dt.date(2024, 5, 1),
                                    end_date=dt.date(2024, 5, 25))
print(df.head())
file = os.path.join(db_path, name + ".tsv")
df.to_csv(file, index=False, sep='\t')

