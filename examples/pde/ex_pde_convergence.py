import datetime as dt
from sdevpy.volatility.localvol import localvol_factory as lvf


# Specify underlying and date
name, valdate = "ABC", dt.datetime(2025, 12, 15)

# Retrieve local volatility
lv = lvf.get_local_vols([name], valdate)[0]

