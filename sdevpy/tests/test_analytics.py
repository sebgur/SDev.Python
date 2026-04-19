import numpy as np
from sdevpy.analytics import black
from sdevpy.utilities.tools import isequal


def test_black_price_straddle():
    """ Straddle = Call + Put """
    expiry, fwd, strike, vol = 1.0, 100.0, 105.0, 0.25
    call = black.price(expiry, strike, True, fwd, vol)
    put = black.price(expiry, strike, False, fwd, vol)
    test = call + put
    ref = 20.77769151258871
    assert isequal(test, ref)

def test_black_price_atm():
    """ Check things are ok at ATM """
    expiry, fwd, strike, vol = 1.0, 100.0, 100.0, 0.25
    test = black.price(expiry, strike, True, fwd, vol)
    ref = 9.94764496602258
    assert isequal(test, ref)


def test_black_price_round_trip():
    """ Round trip between price and implied vol """
    expiry, fwd, vol = 0.5, 100.0,  0.25
    strikes = np.asarray([95, 100.0, 105.0])
    p = black.price(expiry, strikes, True, fwd, vol)
    test = black.implied_vols(expiry, strikes, True, fwd, p)
    ref = np.asarray([0.25000382, 0.24999664, 0.25000281])
    assert np.allclose(test, ref, 1e-6)


def test_black_price_round_trip_newton():
    """ Round trip between price and implied vol using vectorized Newton """
    expiry, fwd, vol = 0.5, 100.0, 0.25
    strikes = np.asarray([95, 100.0, 105.0])
    p = black.price(expiry, strikes, True, fwd, vol)
    test = black.implied_vol_newton(expiry, strikes, True, fwd, p)
    ref = np.asarray([0.25, 0.25, 0.25])
    assert np.allclose(test, ref, 1e-6)


def test_black_price_vectorized():
    """ Check Black price() is fully vectorized """
    expiry = np.asarray([0.5, 1.0, 1.5])
    fwd = np.asarray([100.0, 100.0, 100.0])
    strike = np.asarray([95, 100, 105])
    vol = np.asarray([0.22, 0.25, 0.30])
    is_call = np.asarray([True, True, True])
    test = black.price(expiry, strike, is_call, fwd, vol)
    ref = np.asarray([8.86974195, 9.94764497, 12.57046567])
    assert np.allclose(test, ref, 1e-10)


if __name__ == "__main__":
    test_black_price_vectorized()
