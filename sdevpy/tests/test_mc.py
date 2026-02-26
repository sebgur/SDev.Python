import numpy as np
import datetime as dt
from sdevpy.montecarlo.payoffs.basic import *
from sdevpy.montecarlo.payoffs.vanillas import VanillaOption, VanillaOptionPayoff
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.tools import book as bk
from sdevpy.montecarlo.MonteCarloPricer import price_book


def test_mc():
    valdate = dt.datetime(2025, 12, 15)

    # Create portfolio
    book = bk.Book()
    trades = []
    expiry = dt.datetime(2026, 12, 15)
    trades.append(Trade(VanillaOption('ABC', 100.0, 'Call', expiry), name="vanilla"))
    trades.append(Trade(BasketOption(['XYZ', 'KLM'], [0.5, 0.1], 100.0, 'Call', expiry), name="basket"))
    trades.append(Trade(AsianOption('ABC', 100.0, 'Call'), name="asian"))
    trades.append(Trade(WorstOfBarrier(['ABC', 'XYZ'], 100.0, 'Call', 35.0), name="worstof"))
    book.add_trades(trades)

    # Price
    mc_price = price_book(valdate, book, scramble=False, constr_type='brownianbridge',
                          rng_type='sobol', n_paths=2000)
    test = mc_price['pv']
    ref = np.asarray([8.811443508, 0.0, 4.890335672, 0.006843482])
    # ref = np.asarray([8.919713310, 0.0, 5.001780666, 0.008079992])

    assert np.allclose(test, ref, 1e-8)
