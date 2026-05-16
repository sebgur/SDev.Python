import numpy as np
import pytest
from sdevpy.analytics.americantree import option_price
from sdevpy.analytics import black


TTM = 1.0
SPOT = 100.0
STRIKE = 100.0
VOL = 0.20
RF_RATE = 0.05
DIV_RATE = 0.02
DISC_RATE = 0.05
N_CONV = 200   # steps for convergence tests
N_FAST = 50    # steps for structural/property tests


def _cf_price(is_call):
    """ Black-Scholes closed-form reference (present value). """
    fwd = SPOT * np.exp((RF_RATE - DIV_RATE) * TTM)
    df = np.exp(-DISC_RATE * TTM)
    return df * black.price(TTM, STRIKE, is_call, fwd, VOL)


@pytest.mark.parametrize("method", ["binomial", "trinomial"])
def test_european_call_converges_to_black(method):
    """ European call price must converge to Black-Scholes within 2%. """
    cf = _cf_price(True)
    tree = option_price(TTM, STRIKE, True, False, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_CONV)
    assert abs(tree / cf - 1.0) < 0.02


@pytest.mark.parametrize("method", ["binomial", "trinomial"])
def test_european_put_converges_to_black(method):
    """ European put price must converge to Black-Scholes within 2%. """
    cf = _cf_price(False)
    tree = option_price(TTM, STRIKE, False, False, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_CONV)
    assert abs(tree / cf - 1.0) < 0.02


@pytest.mark.parametrize("method", ["binomial", "trinomial"])
def test_put_call_parity_european(method):
    """ European call - put == discounted(fwd - strike). """
    call = option_price(TTM, STRIKE, True, False, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_CONV)
    put = option_price(TTM, STRIKE, False, False, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_CONV)
    fwd = SPOT * np.exp((RF_RATE - DIV_RATE) * TTM)
    df = np.exp(-DISC_RATE * TTM)
    rhs = df * (fwd - STRIKE)
    assert abs(call - put - rhs) < 0.10


@pytest.mark.parametrize("method", ["binomial", "trinomial"])
def test_american_put_ge_european_put(method):
    """ American put must be >= European put (early-exercise premium is non-negative). """
    eur = option_price(TTM, STRIKE, False, False, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_FAST)
    amer = option_price(TTM, STRIKE, False, True, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_FAST)
    assert amer >= eur - 1e-10


@pytest.mark.parametrize("method", ["binomial", "trinomial"])
def test_american_call_equals_european_call_no_dividends(method):
    """ With div_rate=0, early exercise of a call is never optimal: American == European. """
    eur = option_price(TTM, STRIKE, True, False, SPOT, VOL, RF_RATE, 0.0, DISC_RATE, method, N_CONV)
    amer = option_price(TTM, STRIKE, True, True, SPOT, VOL, RF_RATE, 0.0, DISC_RATE, method, N_CONV)
    assert abs(amer - eur) < 0.01


@pytest.mark.parametrize("method", ["binomial", "trinomial"])
def test_american_put_ge_intrinsic_value(method):
    """ American put must be >= max(K - S, 0) (intrinsic value floor). """
    put = option_price(TTM, STRIKE, False, True, SPOT, VOL, RF_RATE, DIV_RATE, DISC_RATE, method, N_FAST)
    intrinsic = max(STRIKE - SPOT, 0.0)
    assert put >= intrinsic - 1e-10


if __name__ == "__main__":
    print("Hello")
