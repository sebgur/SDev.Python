import numpy as np
import datetime as dt
import pandas as pd
from sdevpy.montecarlo.payoffs.basic import Trade, Instrument, Variance
from sdevpy.montecarlo.mcpricer import price_book
from sdevpy.tools import timegrids
from sdevpy.tools import book as bk
from sdevpy.montecarlo.payoffs import cashflows as cfl
from sdevpy.tools.scalendar import make_schedule
from sdevpy.market import fixings as fxgs
from sdevpy.market.yieldcurve import get_yieldcurve


valdate = dt.datetime(2025, 12, 15)

# Create portfolio
book = bk.Book()
trades = []
start_date = dt.datetime(2025, 11, 15)
expiry = dt.datetime(2026, 12, 15)

# VarSwap
index = Variance('ABC', start_date, expiry)
payoff = index
cf = cfl.Cashflow(payoff, expiry)
trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

# Create book
book.add_trades(trades)

# Price book
mc_price = price_book(valdate, book, constr_type='brownianbridge', rng_type='sobol',
                      n_paths=10000, n_timesteps=50)

# View prices
for i in range(len(mc_price['pv'])):
    print("MC:", mc_price['pv'][i])

# Check values
alldates = make_schedule("USD", start_date, expiry, "1D")
hist_fixing_dates = []
for date in alldates:
    if date < valdate:
        hist_fixing_dates.append(date)

# Fetch historical fixings
hist_fixings = fxgs.get_fixings('ABC', hist_fixing_dates)
df = pd.DataFrame({'Date': hist_fixing_dates, 'Fixing': hist_fixings})
print("Hist data", df)

# Hist variance
current_sum = 0.0
current_var = 0.0
if hist_fixings is not None and len(hist_fixings) > 1:
    log_returns = np.diff(np.log(np.asarray(hist_fixings)))
    log2 = np.power(log_returns, 2)
    current_sum = log2.sum()

n_returns = len(hist_fixings) - 1
current_var = current_sum / n_returns * 252 * 10000
current_vol = np.sqrt(current_var)
print(f"current sum: {current_sum}")
print(f"current var: {current_var}")
print(f"current vol: {current_vol}")
print(f"n returns: {n_returns}")
print(f"check: {len(log_returns)}")

spot = 100
current_fixing = hist_fixings[-1]
current_inc = np.log(spot / current_fixing)
current_inc = np.power(current_inc, 2)
print(f"current fixing: {current_fixing}")
print(f"current inc: {current_inc}")

disc_curve = get_yieldcurve("USD.SOFR.1D", valdate)
df = disc_curve.discount(expiry)
n_returns = 269
current_pv = df * current_sum / n_returns * 252 * 10000
inc_pv = df * current_inc / n_returns * 252 * 10000
print(f"current pv: {current_pv}")
print(f"inc pv: {inc_pv}")
print(f"pv: {current_pv + inc_pv}")
print(f"mc vol: {np.sqrt(mc_price['pv'][0] / df)}")
