import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sdevpy.pricingcontext import default_calibration_repository
from sdevpy.utilities import timegrids
from sdevpy.utilities import dates as dts


valdate = dt.datetime(2025, 12, 15)
curve_id = "USD.SOFR.1D"
pd_date_fmt = {'Date': lambda d: d.strftime('%d-%b-%Y')}

# Retrieve curve from repository
calib = default_calibration_repository()
ds = calib[valdate]
curve = ds.get_yieldcurve(curve_id)

# Test discount at valdate (Claude says there's an issue)
print(f"T0 discount: {curve.discount(valdate)}")

# View calibration pillars
dates, dfs = curve.dates, curve.dfs
times = timegrids.model_time(valdate, dates)
zrs = -np.log(dfs) / times
pillar_df = pd.DataFrame({'Date': dates, 'Time': times, 'ZR': zrs, 'DF': dfs})
print()
print("Calibration pillars")
print(pillar_df.to_string(index=False, formatters=pd_date_fmt))

# View curve interpolation against calibration pillars
interp_days = [1, 2, 3, 4, 5]
interp_days.extend([10 * (i + 1) for i in range(1500)])
interp_dates = [dts.advance_int(valdate, days=d) for d in interp_days]
interp_times = timegrids.model_time(valdate, interp_dates)
interp_dfs = curve.discount(interp_dates)
interp_zrs = -np.log(interp_dfs) / interp_times
interp_df = pd.DataFrame({'Date': interp_dates, 'Time': interp_times, 'ZR': interp_zrs, 'DF': interp_dfs})
print()
print("Interpolated curve")
print(interp_df.head(10).to_string(index=False, formatters=pd_date_fmt))

# Plot
plot_start, plot_end = dt.datetime(2052, 12, 15), dt.datetime(2065, 5, 15) # Early part
plot_pillar_df = pillar_df[pillar_df['Date'].between(plot_start, plot_end)]
plot_interp_df = interp_df[interp_df['Date'].between(plot_start, plot_end)]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(plot_pillar_df['Date'], plot_pillar_df['DF'])
axs[0].plot(plot_interp_df['Date'], plot_interp_df['DF'])
axs[0].set_title('Discount factors')

axs[1].scatter(plot_pillar_df['Date'], plot_pillar_df['ZR'])
axs[1].plot(plot_interp_df['Date'], plot_interp_df['ZR'])
axs[1].set_title('Zero-rates')

plt.tight_layout()
plt.show()
