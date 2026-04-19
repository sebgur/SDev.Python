import numpy as np
from sdevpy.utilities.tools import isequal
from sdevpy.volatility.impliedvol.models import svi, biexp, cubicvol, vsvi, gsvi
from sdevpy.volatility.impliedvol.impliedvol_calib import TsIvObjectiveBuilder
from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
from sdevpy.volatility.impliedvol.models.tssvi2 import TsSvi2
from sdevpy.volatility.impliedvol.models.logmix import LogMix
from sdevpy.volatility.impliedvol.models import sabr


def test_sabr():
    # Test near ATM
    expiry = 0.5
    fwd = 0.04
    params = {'LnVol': 0.25, 'Beta': 0.4, 'Nu': 0.50, 'Rho': -0.25}
    strikes = np.asarray([0.01, 0.04, 0.06])
    test = sabr.sabr_from_dict(expiry, strikes, fwd, params)
    ref = np.asarray([0.54225604, 0.25208659, 0.22711695])
    assert np.allclose(test, ref, 1e-10)


def test_logmix_objective():
    surface = LogMix(2)
    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90., 100., 110.])
    f = np.asarray([95., 105., 115.])
    mkt_vols = np.asarray([0.30, 0.25, 0.20])
    mkt_prices = np.asarray([10.30, 8.25, 5.20])
    params = surface.initial_point()
    builder = TsIvObjectiveBuilder(surface, t, k, f, mkt_vols, mkt_prices)
    test = builder.objective(params)
    assert isequal(test, 7.268371270480672)


def test_logmix():
    surface = LogMix(3)
    params = surface.initial_point()
    surface.update_params(params)

    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90, 100, 110])
    f = np.asarray([95, 105, 115])
    is_call = True
    test = surface.calculate(t, k, is_call, f)
    ref = np.asarray([8.09016928, 12.68788175, 16.77192795])
    assert np.allclose(test, ref, 1e-10)


def test_tssvi2_objective():
    surface = TsSvi2()
    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90., 100., 110.])
    f = np.asarray([95., 105., 115.])
    mkt_vols = np.asarray([0.30, 0.25, 0.20])
    mkt_prices = None
    params = surface.initial_point()
    # params = [0.20, 0.25, 0.10, 2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    builder = TsIvObjectiveBuilder(surface, t, k, f, mkt_vols, mkt_prices)
    test = builder.objective(params)
    assert isequal(test, 0.065354183239)


def test_tssvi2():
    surface = TsSvi2()
    params = surface.initial_point()
    # params = [0.20, 0.25, 0.10, 2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    surface.update_params(params)

    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90, 100, 110])
    f = np.asarray([95, 105, 115])
    is_call = True
    test = surface.calculate(t, k, is_call, f)
    ref = np.asarray([0.41214181, 0.23553742, 0.20534387])
    assert np.allclose(test, ref, 1e-10)


def test_tssvi1_objective():
    surface = TsSvi1()
    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90., 100., 110.])
    f = np.asarray([95., 105., 115.])
    mkt_vols = np.asarray([0.30, 0.25, 0.20])
    mkt_prices = None
    params = [0.20, 0.25, 0.10, 2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    builder = TsIvObjectiveBuilder(surface, t, k, f, mkt_vols, mkt_prices)
    test = builder.objective(params)
    assert isequal(test, 0.0997965850768)


def test_tssvi1():
    surface = TsSvi1()
    params = [0.20, 0.25, 0.10, 2.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    surface.update_params(params)

    t = np.asarray([0.5, 1.5, 2.5])
    k = np.asarray([90, 100, 110])
    f = np.asarray([95, 105, 115])
    is_call = True
    test = surface.calculate(t, k, is_call, f)
    ref = np.asarray([0.28683819, 0.32803753, 0.35367168])
    assert np.allclose(test, ref, 1e-10)


def test_svi_formula():
    t = 0.5
    params = svi.sample_params(t)
    params[1] = 0.1 # b
    params[2] = -0.30 # rho
    params[4] = 0.2 # sigma
    m = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(m) # Log-moneyness

    test = svi.svi_formula(t, log_m, params)
    ref = np.asarray([0.49837104, 0.32015621, 0.40644314])
    assert np.allclose(test, ref, 1e-10)


def test_biexp_formula():
    t = 0.5
    params = biexp.sample_params(t)
    m = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(m) # Log-moneyness

    test = biexp.biexp_formula(t, log_m, params)
    ref = np.asarray([0.29982632, 0.25, 0.27999992])
    assert np.allclose(test, ref, 1e-10)


def test_cubicvol_formula():
    t = 1.5
    atm, skew, kurt, vl, vr = 0.25, 0.1, 0.25, 0.30, 0.27
    m = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(m) # Log-moneyness

    test = cubicvol.cubicvol(t, log_m, atm, skew, kurt, vl, vr)
    ref = np.asarray([0.3, 0.25, 0.249886])
    assert np.allclose(test, ref, 1e-10)


def test_vsvi_formula():
    t = 1.5
    base_vol = 0.25
    a, b, rho, m, sigma = base_vol, 0.1 / t, -0.25, 0.0, 0.25 * t # a, b, rho, m, sigma
    mx = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(mx) # Log-moneyness

    test = vsvi.vsvi(log_m, a, b, rho, m, sigma)
    ref = np.asarray([0.31409145, 0.275, 0.29098655])
    assert np.allclose(test, ref, 1e-10)


def test_gsvi_formula():
    base_vol = 0.25
    a, b, rho, m, sigma = base_vol, 0.1, -0.25, 0.0, 0.25 # a, b, rho, m, sigma
    mx = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(mx) # Log-moneyness

    test = gsvi.gsvi_formula(log_m, [a, b, rho, m, sigma])
    ref = np.asarray([0.58396406, 0.52440442, 0.55349496])
    assert np.allclose(test, ref, 1e-10)


if __name__ == "__main__":
    test_sabr()
    # test_tssvi1_objective()
    # test_tssvi1()
    # test_svi_formula()
    # test_biexp_formula()
    # test_cubicvol_formula()
    # test_vsvi_formula()
    # test_gsvi_formula()
