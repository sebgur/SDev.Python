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
from sdevpy.market.yieldcurve import get_yieldcurve


#################### TODO #########################################################################
# * Asian: add averaging window to even dates
# * Calculate payoff indexes on event date grid?
# * Multi-cashflow design
# * Introduce concept of past fixings
# * Implement var swap spread payoff
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
    expiry = dt.datetime(2026, 12, 15)
    v_name, v_strike, v_type = 'ABC', 100.0, 'Call' # For check against CF
    trades.append(Trade(VanillaOption(v_name, v_strike, v_type, expiry), name="vanilla"))
    trades.append(Trade(BasketOption(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call', expiry), name="basket"))
    trades.append(Trade(AsianOption('ABC', 100.0, 'Call'), name="asian"))
    trades.append(Trade(WorstOfBarrier(['ABC', 'XYZ'], 100.0, 'Call', 35.0), name="worstof"))
    book.add_trades(trades)

    # Price book
    mc_price = price_book(valdate, book, constr_type='brownianbridge', rng_type='sobol',
                          n_paths=4, n_timesteps=5)
                        #   n_paths=100*1000, n_timesteps=50)
    # print(mc_price)

    # Gather all names in the book
    names = book.names
    print(f"Book names: {names}")
    print(f"Number of assets: {len(names)}")

    # Closed-form for vanilla
    disc_curve = get_yieldcurve(book.csa_curve_id, valdate)
    fwd_curves = get_forward_curves(names, valdate)
    lvs = get_local_vols(names, valdate)
    eventdates = get_eventdates(book)
    event_tgrid = np.array([timegrids.model_time(valdate, date) for date in eventdates])
    dmax = eventdates.max()
    T = event_tgrid[-1]
    name_idx = names.index(v_name)
    fwd = fwd_curves[name_idx].value_float(T)
    df = disc_curve.discount(dmax)
    # print(df)
    sigma = np.asarray([0.2] * len(names))
    cf_price = df * black.price(T, v_strike, v_type, fwd, sigma[name_idx])

    print("MC:", mc_price['pv'][0])
    print("MC:", mc_price['pv'][1])
    print("MC:", mc_price['pv'][2])
    print("MC:", mc_price['pv'][3])
    print("CF:", cf_price)
