import numpy as np
import datetime as dt
import logging
from sdevpy.montecarlo.payoffs.basic import Trade, Instrument, Variance
from sdevpy.montecarlo.payoffs.vanillas import VanillaOption
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.montecarlo.mcpricer import price_book, get_local_vols
from sdevpy.analytics import black
from sdevpy.tools import timegrids
from sdevpy.tools import book as bk
from sdevpy.market.yieldcurve import get_yieldcurve
from sdevpy.market.eqforward import get_forward_curves
from sdevpy.montecarlo.payoffs import cashflows as cfl
logger = logging.getLogger("mcrun")
logger.setLevel(logging.DEBUG)


#################### TODO #########################################################################
# * Finish varswap payoff, Check var swap values, Add volswaps
# * Check accuracy against LV calib
# * Calculate vega through LV calib (save BM/interpolation)

# * Implement no-arb time parametric IVs (mixture of lognormals, SVI)
# * Implementing exact Dupire through python vectorization and try parallelization
# * Implement DE with parallelization on the population


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)

    logger.debug("some debug")
    logger.info("some info")
    logger.warning("some warning")
    logger.error("some error")
    logger.critical("some critical")

    # Create portfolio
    book = bk.Book()
    trades = []
    start_date = dt.datetime(2025, 11, 15)
    expiry = dt.datetime(2026, 12, 15)
    expiry2 = dt.datetime(2027, 12, 15)
    v_name, v_strike, v_type = 'ABC', 100.0, 'Call' # For check against CF

    # Vanilla
    index = VanillaOption(v_name, v_strike, v_type, expiry)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Basket option
    index = BasketOption(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call', expiry)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Asian option
    index = AsianOption('ABC', 100.0, 'Call', valdate, expiry, freq='5D')
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Worst-of barrier
    index = WorstOfBarrier(['ABC', 'XYZ'], 100.0, 'Call', 35.0)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # VarSwap
    index = Variance('ABC', start_date, expiry)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Create book
    book.add_trades(trades)

    # Price book
    mc_price = price_book(valdate, book, constr_type='brownianbridge', rng_type='sobol',
                          n_paths=10000, n_timesteps=50)

    # Gather all names in the book
    names = book.names
    print(f"Book names: {names}")
    print(f"Number of assets: {len(names)}")

    # Closed-form for vanilla
    disc_curve = get_yieldcurve(book.csa_curve_id, valdate)
    fwd_curves = get_forward_curves(names, valdate)
    lvs = get_local_vols(names, valdate)
    name_idx = names.index(v_name)
    fwd = fwd_curves[name_idx].value(expiry)
    df = disc_curve.discount(expiry)
    ttm = timegrids.model_time(valdate, expiry)
    sigma = np.asarray([0.2] * len(names))
    cf_price = df * black.price(ttm, v_strike, v_type, fwd, sigma[name_idx])

    for i in range(len(mc_price['pv'])):
        print("MC:", mc_price['pv'][i])

    print("CF:", cf_price)
