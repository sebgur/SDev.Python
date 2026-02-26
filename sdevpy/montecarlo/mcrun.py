import numpy as np
import datetime as dt
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.PathGenerator import PathGenerator
from sdevpy.montecarlo.payoffs.basic import *
from sdevpy.montecarlo.payoffs.vanillas import VanillaOption, VanillaOptionPayoff
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.montecarlo.MonteCarloPricer import *
from sdevpy.models import localvol_factory as lvf
from sdevpy.analytics import black
from sdevpy.tools import timegrids, timer
from sdevpy.tools import book as bk


#################### TODO #########################################################################
# * Handle event dates: check what happens if maturity date (as event) is last on disc. grid
# * Introduce discount curve and discount at cash-flow payment times
# * Implement simple forward curve as linear interpolation of surface's forwards
# * Introduce concept of past fixings
# * Implement var swap spread payoff
# * Introduce multi-cash-flow payoffs
# * Check accuracy against LV calib
# * Calculate vega through LV calib
# * Greeks by saving BM and maybe time interpolation too?

# * Implement no-arb time parametric IVs (mixture of lognormals, SVI)
# * Implementing exact Dupire through python vectorization and try parallelization
# * Implement DE with parallelization on the population
# * Mistral: use numba JIT, parallelization (joblib, Ray)


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)

    # Create portfolio
    book = bk.Book()
    trades = []
    v_name, v_strike, v_type = 'ABC', 100.0, 'Call' # For check against CF
    trades.append(Trade(VanillaOption(v_name, v_strike, v_type), name="vanilla"))
    trades.append(Trade(BasketOption(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call'), name="basket"))
    trades.append(Trade(AsianOption('ABC', 100.0, 'Call'), name="asian"))
    trades.append(Trade(WorstOfBarrier(['ABC', 'XYZ'], 100.0, 'Call', 35.0), name="worstof"))
    book.add_trades(trades)

    # Gather all names in the book
    names = book.names
    print(f"Book names: {names}")
    print(f"Number of assets: {len(names)}")

    # Price book
    mc_price = price_book(valdate, book)
    print(mc_price)

    # Closed-form for vanilla
    df = 0.90
    fwd_curves = get_forward_curves(names, valdate)
    lvs, sigma = get_local_vols(names, valdate)
    eventdates = get_eventdates(book, valdate)
    event_tgrid = np.array([timegrids.model_time(valdate, date) for date in eventdates])
    T = event_tgrid[-1]
    name_idx = names.index(v_name)
    fwd = fwd_curves[name_idx](T)
    cf_price = df * black.price(T, v_strike, v_type, fwd, sigma[name_idx])

    print("MC:", mc_price['pv'][0])
    print("CF:", cf_price)

    # timer_path.print()
    # timer_mc.print()
