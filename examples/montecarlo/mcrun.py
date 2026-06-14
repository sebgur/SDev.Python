import numpy as np
import datetime as dt
from sdevpy.montecarlo.payoffs.basic import Trade, Instrument, Variance
from sdevpy.montecarlo.payoffs.vanillas import make_vanilla_option
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, make_basket_option, make_asian_option
from sdevpy.montecarlo.payoffs import cashflows as cfl
from sdevpy.montecarlo.mcpricer import price_book
from sdevpy.analytics import black
from sdevpy.utilities import timegrids
from sdevpy.utilities import book as bk
from sdevpy.market import provider as mdp
from sdevpy.pricingcontext import default_pricing_context
from sdevpy import logger
logger.configure()


#################### TODO #########################################################################
# * Make notebook with varswap and volswap trades pricing in MC from scratch (market data)
    ## Varswap ##
    # cash-flow = N_vega / (2 * strike) * (variance - strike^2)
    ## Volswap ##
    # vol = sqrt(variance)
    # cashflow = N_vega * (vol - strike)


if __name__ == "__main__":
    valdate = dt.datetime(2025, 12, 15)
    ctx = default_pricing_context()

    # Create portfolio
    book = bk.Book()
    trades = []
    start_date = dt.datetime(2025, 11, 15)
    expiry = dt.datetime(2026, 12, 15)
    expiry2 = dt.datetime(2027, 12, 15)
    v_name, v_strike, v_type = 'ABC', 100.0, 'Call' # For check against CF

    # Vanilla
    index = make_vanilla_option(v_name, v_strike, v_type, expiry)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Basket option
    index = make_basket_option(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call', expiry)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Asian option
    index = make_asian_option('ABC', 100.0, 'Call', valdate, expiry, freq='5D')
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Worst-of barrier
    index = WorstOfBarrier(['ABC', 'XYZ'], expiry, 100.0, 'Call', 35.0)
    cf = cfl.Cashflow(index, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # VarSwap
    index = Variance('ABC', start_date, expiry)
    vstrike = 14.0
    payoff = index - vstrike * vstrike
    cf = cfl.Cashflow(payoff, expiry)
    trades.append(Trade(Instrument(cashflow_legs=[[cf]])))

    # Create book
    book.add_trades(trades)

    # Price book
    mc_price = price_book(valdate, book, ctx, constr_type='brownianbridge', rng_type='sobol',
                          n_paths=10000, n_timesteps=50)

    # Gather all names in the book
    names = book.names
    print(f"Book names: {names}")
    print(f"Number of assets: {len(names)}")

    # Closed-form for vanilla
    md_prov = ctx.market_provider
    disc_curve = md_prov.get_yieldcurve(book.csa_curve_id, valdate)
    fwd_curves = mdp.get_eq_forward_curves(names, valdate, md_prov)
    name_idx = names.index(v_name)
    fwd = fwd_curves[name_idx].value(expiry)
    df = disc_curve.discount(expiry)
    ttm = timegrids.model_time(valdate, expiry)
    sigma = np.asarray([0.2] * len(names))
    cf_price = df * black.price(ttm, v_strike, v_type, fwd, sigma[name_idx])

    for i in range(len(mc_price['pv'])):
        print("MC:", mc_price['pv'][i])

    print("CF:", cf_price)
