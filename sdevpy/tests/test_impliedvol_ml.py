import numpy as np
from sdevpy.volatility.impliedvol.models import fbsabr
from sdevpy.volatility.impliedvol.models import mcheston
from sdevpy.volatility.impliedvol.models import mcsabr
from sdevpy.volatility.impliedvol.models import mczabr


FWD = 0.05

############# MC FBSABR ###########################################################################
FBSABR_PARAMS = {'LnVol': 0.30, 'Beta': 0.5, 'Nu': 0.50, 'Rho': 0.0}


def test_mcfbsabr_calculate_alpha():
    result = fbsabr.calculate_fbsabr_alpha(0.30, 0.05, 0.5)
    expected = 0.30 * (0.05 ** 0.5)
    assert np.isclose(result, expected)


def test_mcfbsabr_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = fbsabr.price(expiries, strikes, are_calls, FWD, FBSABR_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mcfbsabr_euler_scheme():
    expiries = np.asarray([1.0])
    strikes = [[FWD]]
    are_calls = [[True]]
    result = fbsabr.price(expiries, strikes, are_calls, FWD, FBSABR_PARAMS, num_mc=10, points_per_year=5,
                   scheme='Euler')
    assert result.shape == (1, 1)
    assert result[0, 0] >= 0.0


############# MC HESTON ###########################################################################
HESTON_PARAMS = {'LnVol': 0.25, 'Kappa': 1.0, 'Theta': 0.0625, 'Xi': 0.50, 'Rho': -0.25}


def test_mcheston_calculate_v0():
    assert np.isclose(mcheston.calculate_v0(0.25), 0.0625)


def test_mcheston_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = mcheston.price(expiries, strikes, are_calls, FWD, HESTON_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mcheston_price_nonnegative():
    expiries = np.asarray([1.0])
    strikes = [[FWD * 0.8, FWD, FWD * 1.2]]
    are_calls = [[True, True, True]]
    result = mcheston.price(expiries, strikes, are_calls, FWD, HESTON_PARAMS, num_mc=10, points_per_year=5)
    assert np.all(result >= 0.0)


############# MC SABR ###########################################################################
MCSABR_PARAMS = {'LnVol': 0.25, 'Beta': 0.5, 'Nu': 0.50, 'Rho': -0.25}


def test_mcsabr_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = mcsabr.price(expiries, strikes, are_calls, FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mcsabr_price_nonnegative():
    expiries = np.asarray([1.0])
    strikes = [[FWD * 0.8, FWD, FWD * 1.2]]
    are_calls = [[True, True, True]]
    result = mcsabr.price(expiries, strikes, are_calls, FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5)
    assert np.all(result >= 0.0)


def test_mcsabr_log_euler_scheme():
    expiries = np.asarray([1.0])
    result = mcsabr.price(expiries, [[FWD]], [[True]], FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5,
                   scheme='LogEuler')
    assert result.shape == (1, 1)
    assert result[0, 0] >= 0.0


def test_mcsabr_andersen_scheme():
    expiries = np.asarray([1.0])
    result = mcsabr.price(expiries, [[FWD]], [[True]], FWD, MCSABR_PARAMS, num_mc=10, points_per_year=5,
                   scheme='Andersen')
    assert result.shape == (1, 1)
    assert result[0, 0] >= 0.0


############# MC ZABR ###########################################################################
MCZABR_PARAMS = {'LnVol': 0.25, 'Beta': 0.7, 'Nu': 0.47, 'Rho': -0.48, 'Gamma': 1.0}


def test_mczabr_calculate_alpha():
    result = mczabr.calculate_zabr_alpha(0.25, 0.05, 0.7)
    expected = 0.25 * (0.05 ** 0.3)
    assert np.isclose(result, expected)
    # beta=1: alpha = ln_vol * fwd^0 = ln_vol
    assert np.isclose(mczabr.calculate_zabr_alpha(0.25, 0.05, 1.0), 0.25)


def test_mczabr_price_output_shape():
    expiries = np.asarray([0.5, 1.0])
    strikes = [[FWD * 0.9, FWD, FWD * 1.1]] * 2
    are_calls = [[True, True, True]] * 2
    result = mczabr.price(expiries, strikes, are_calls, FWD, MCZABR_PARAMS, num_mc=10, points_per_year=5)
    assert result.shape == (2, 3)


def test_mczabr_price_nonnegative():
    expiries = np.asarray([1.0])
    strikes = [[FWD * 0.8, FWD, FWD * 1.2]]
    are_calls = [[True, True, True]]
    result = mczabr.price(expiries, strikes, are_calls, FWD, MCZABR_PARAMS, num_mc=10, points_per_year=5)
    assert np.all(result >= 0.0)


def test_mczabr_price_gamma_one_close_to_mcsabr():
    # ZABR with gamma=1 reduces to SABR (LogEuler scheme) — prices should be close
    sabr_params = {k: v for k, v in MCZABR_PARAMS.items() if k != 'Gamma'}
    expiries = np.asarray([1.0])
    strikes = [[FWD]]
    are_calls = [[True]]
    z = mczabr.price(expiries, strikes, are_calls, FWD, MCZABR_PARAMS, num_mc=2000, points_per_year=10)
    s = mcsabr.price(expiries, strikes, are_calls, FWD, sabr_params, num_mc=2000, points_per_year=10,
                   scheme='LogEuler')
    assert np.abs(z[0, 0] - s[0, 0]) < 5e-4  # within 0.5bp forward
