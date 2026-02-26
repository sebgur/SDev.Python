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
    mc_price = price_book(valdate, book)
    test = mc_price['pv']
    # ref = np.asarray([8.185738996, 0, 4.617361664, 0.000993840])
    ref = np.asarray([8.937606441, 0, 5.041470461, 0.001085125])

    assert np.allclose(test, ref, 1e-8)
