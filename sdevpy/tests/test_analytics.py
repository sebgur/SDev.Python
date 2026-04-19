import numpy as np
from sdevpy.analytics import black
from sdevpy.analytics import bachelier
from sdevpy.utilities.tools import isequal

############ Black ################################################################################

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


def test_black_price_vectorized():
    """ Check price is fully vectorized """
    expiry = np.asarray([0.5, 1.0, 1.5])
    fwd = np.asarray([100.0, 100.0, 100.0])
    strike = np.asarray([95, 100, 105])
    vol = np.asarray([0.22, 0.25, 0.30])
    is_call = np.asarray([True, True, True])
    test = black.price(expiry, strike, is_call, fwd, vol)
    ref = np.asarray([8.86974195, 9.94764497, 12.57046567])
    assert np.allclose(test, ref, 1e-10)


def test_black_roundtrip():
    """ Round trip via Brent (semi-vectorized on strike) """
    expiry, fwd, vol = 0.5, 100.0,  0.25
    strikes = np.asarray([95, 100.0, 105.0])
    p = black.price(expiry, strikes, True, fwd, vol)
    test = black.implied_vols(expiry, strikes, True, fwd, p)
    ref = np.asarray([0.25000382, 0.24999664, 0.25000281])
    assert np.allclose(test, ref, 1e-6)


def test_black_roundtrip_newton():
    """ Round trip via Newton (vectorized on strike) """
    expiry, fwd, vol = 0.5, 100.0, 0.25
    strikes = np.asarray([95, 100.0, 105.0])
    p = black.price(expiry, strikes, True, fwd, vol)
    test = black.implied_vol_newton(expiry, strikes, True, fwd, p)
    ref = np.asarray([0.25, 0.25, 0.25])
    assert np.allclose(test, ref, 1e-6)


############ Bachelier ############################################################################

def test_bachelier_price_straddle():
    """ Straddle = Call + Put """
    expiry, fwd, strike, vol = 1.0, 0.04, 0.045, 0.05
    call = bachelier.price(expiry, strike, True, fwd, vol)
    put = bachelier.price(expiry, strike, False, fwd, vol)
    test = call + put
    ref = 0.040093533120471
    assert isequal(test, ref)


def test_bachelier_price_atm():
    """ Check things are ok at ATM """
    expiry, fwd, strike, vol = 1.0, 0.04, 0.04, 0.05
    test = bachelier.price(expiry, strike, True, fwd, vol)
    ref = 0.019947114020071637
    assert isequal(test, ref)


def test_bachelier_price_vectorized():
    """ Check price is fully vectorized """
    expiry = np.asarray([0.5, 1.5, 2.5])
    vol = np.asarray([0.05, 0.05, 0.05])
    fwd = np.asarray([0.04, 0.04, 0.04])
    strike = np.asarray([0.035, 0.04, 0.045])
    is_call = np.asarray([True, True, False])
    test = bachelier.price(expiry, strike, is_call, fwd, vol)
    ref = np.asarray([0.01674555, 0.02443013, 0.03410221])
    assert np.allclose(test, ref, 1e-10)


def test_bachelier_roundtrip():
    """ Round-trip via Jaeckel (vectorized) """
    expiry = np.asarray([2.0, 1.5, 2.0, 2.0])
    fwd = np.asarray([0.04, 0.04, 0.04, 0.04])
    vol = np.asarray([0.06, 0.06, 0.05, 0.06])
    strikes = np.asarray([0.02, 0.03, 0.04, 0.05])
    is_call = np.asarray([True, True, True, False])
    prices = bachelier.price(expiry, strikes, is_call, fwd, vol)
    test = bachelier.implied_vol(expiry, strikes, is_call, fwd, prices)
    ref = np.asarray([0.06, 0.06, 0.05, 0.06])
    assert np.allclose(test, ref, 1e-10)


def test_bachelier_roundtrip_jaeckel():
    """ Round-trip via Jaeckel (non-vectorized) """
    expiry, fwd, vol = 2.0, 0.04, 0.06
    strikes = np.asarray([0.02, 0.03, 0.04, 0.05])
    prices = bachelier.price(expiry, strikes, True, fwd, vol)
    test = []
    for strike, price in zip(strikes, prices, strict=True):
        test.append(bachelier.implied_vol_jaeckel(expiry, strike, True, fwd, price))

    test = np.asarray(test)
    ref = np.asarray([0.06, 0.06, 0.06, 0.06])
    assert np.allclose(test, ref, 1e-10)


def test_bachelier_roundtrip_brent():
    """ Round-trip via Brent (non-vectorized) """
    expiry, fwd, vol = 2.0, 0.04, 0.06
    strikes = np.asarray([0.02, 0.03, 0.04, 0.05])
    prices = bachelier.price(expiry, strikes, True, fwd, vol)
    test = []
    for strike, price in zip(strikes, prices, strict=True):
        test.append(bachelier.implied_vol_solve(expiry, strike, True, fwd, price))

    test = np.asarray(test)
    ref = np.asarray([0.06000024396, 0.05999898432, 0.05999999999, 0.05999898432])
    assert np.allclose(test, ref, 1e-10)


if __name__ == "__main__":
    test_bachelier_roundtrip()
