import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sdevpy.pricingcontext import default_calibration_repository
from sdevpy.utilities import timegrids
from sdevpy.utilities import dates as dts


valdate = dt.datetime(2025, 12, 15)
curve_id = "USD.SOFR.1D"

calib = default_calibration_repository()

ds = calib[valdate]

curve = ds.get_yieldcurve(curve_id)

expiry = dt.datetime(2026, 12, 15)

df = curve.discount(expiry)

print(df)

dates, dfs = curve.dates, curve.dfs
times = timegrids.model_time(valdate, dates)
zrs = -np.log(dfs) / times

print(dates)

print(zrs)

view_days = [1, 2, 3, 4, 5]
view_days.extend([10 * (i + 1) for i in range(10)])

view_dates = [dts.advance_int(valdate, days=d) for d in view_days]
view_times = timegrids.model_time(valdate, view_dates)
view_dfs = curve.discount(view_dates)
view_zrs = -np.log(view_dfs) / view_times

print(curve.discount(valdate))


fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(dates, dfs)
axs[0, 0].plot(view_dates, view_dfs)

axs[0, 1].scatter(dates, zrs)
axs[0, 1].plot(view_dates, view_zrs)

plt.show()

