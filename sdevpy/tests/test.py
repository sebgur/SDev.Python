import numpy as np
import datetime as dt
from sdevpy.montecarlo.payoffs.basic import Trade, Instrument, Variance
from sdevpy.montecarlo.mcpricer import price_book
from sdevpy.tools import timegrids
from sdevpy.tools import book as bk
from sdevpy.montecarlo.payoffs import cashflows as cfl


valdate = dt.datetime(2025, 12, 15)

# Create portfolio
book = bk.Book()
trades = []
start_date = dt.datetime(2025, 11, 15)
expiry = dt.datetime(2026, 12, 15)

# VarSwap
index = Variance('ABC', start_date, expiry)
payoff = 10000 * index - 25
cf = cfl.Cashflow(payoff, expiry)
trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

# Create book
book.add_trades(trades)

# Price book
mc_price = price_book(valdate, book, constr_type='brownianbridge', rng_type='sobol',
                      n_paths=10, n_timesteps=50)

# Check

for i in range(len(mc_price['pv'])):
    print("MC:", mc_price['pv'][i])

