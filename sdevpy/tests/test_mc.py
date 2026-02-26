import numpy as np
import datetime as dt
from sdevpy.montecarlo.payoffs.basic import *
from sdevpy.montecarlo.payoffs.vanillas import VanillaOption, VanillaOptionPayoff
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.montecarlo.MonteCarloPricer import price_book


def test_mc():
    valdate = dt.datetime(2025, 12, 15)

    # Create portfolio
    book = []
    v_name, v_strike, v_type = 'ABC', 100.0, 'Call' # For check against CF
    book.append(Trade(VanillaOption(v_name, v_strike, v_type), name="vanilla"))
    book.append(Trade(BasketOption(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call'), name="basket"))
    book.append(Trade(AsianOption('ABC', 100.0, 'Call'), name="asian"))
    book.append(Trade(WorstOfBarrier(['ABC', 'XYZ'], 100.0, 'Call', 35.0), name="worstof"))

    # Gather all names in the book and set their indexes in the instruments
    names = list_names([t.instrument for t in book])
    n_names = len(names)
    for trade in book:
        trade.instrument.set_nameindexes(names)

    mc_price = price_book(valdate, book)
    test = mc_price['pv']
    print(mc_price)
    ref = np.asarray([8.185738996, 0, 4.617361664, 0.000993840])

    assert np.allclose(test, ref, 1e-8)
