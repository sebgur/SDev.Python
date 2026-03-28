import numpy as np
import datetime as dt
from sdevpy.montecarlo.payoffs.basic import Trade, Instrument
from sdevpy.montecarlo.payoffs.vanillas import VanillaOption
from sdevpy.montecarlo.payoffs.exotics import WorstOfBarrier, BasketOption, AsianOption
from sdevpy.tools import book as bk
from sdevpy.montecarlo.MonteCarloPricer import price_book
from sdevpy.montecarlo.MonteCarloPricer import path_interp_coeffs, interp_paths
from sdevpy.montecarlo.payoffs import cashflows as cfl


def test_mc():
    valdate = dt.datetime(2025, 12, 15)

    ## Create portfolio ##
    book = bk.Book()
    trades = []
    expiry = dt.datetime(2026, 12, 15)

    # Vanilla
    index = VanillaOption('ABC', 100.0, 'Call', expiry)
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

    # Create book
    book.add_trades(trades)

    ## Price ##
    mc_price = price_book(valdate, book, scramble=False, constr_type='brownianbridge',
                          rng_type='sobol', n_paths=2000)
    test = mc_price['pv']
    ref = np.asarray([8.811443508, 0.0, 4.90812947, 0.006843482])
    # ref = np.asarray([8.811443508, 0.0, 4.890335672, 0.006843482])
    assert np.allclose(test, ref, 1e-8)


def test_path_interpolation():
    # Discretization
    disc_times = np.asarray([0, 1, 2])
    disc_paths = np.asarray([
        [[100, 10],[110, 11],[120, 12]],
        [[100, 10],[90, 9],[80, 8]],
        [[100, 10],[150, 15],[200, 20]],
        [[100, 10],[70, 7],[60, 6]]])

    # Event dates
    event_times = np.asarray([0.0, 0.2, 1.0, 1.4, 2.0, 2.5])

    # Interpolation
    idx, w0, w1 = path_interp_coeffs(disc_times, event_times)
    test = interp_paths(disc_paths, idx, w0, w1)

    ref = np.asarray([
        [[100., 10.], [102., 10.2], [110., 11.], [114., 11.4], [120., 12.], [125.,   12.5]],
        [[100., 10.], [98., 9.8], [90., 9.], [86., 8.6], [80., 8.], [75., 7.5]],
        [[100., 10.], [110., 11.], [150., 15.], [170., 17.], [200., 20.], [225., 22.5]],
        [[100., 10.], [94., 9.4], [70., 7.], [66., 6.6], [60., 6.], [55., 5.5]]])
    assert np.allclose(test, ref, 1e-8)
