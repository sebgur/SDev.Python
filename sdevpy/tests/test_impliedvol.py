import numpy as np
import datetime as dt
import pytest
from scipy.integrate import quad
from sdevpy.utilities.tools import isequal
from sdevpy.volatility.impliedvol.models import svi, biexp, cubicvol, vsvi
from sdevpy.volatility.impliedvol.impliedvol_calib import TsIvObjectiveBuilder, TsIvCalibrator
from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
from sdevpy.volatility.impliedvol.models.tssvi2 import TsSvi2
from sdevpy.volatility.impliedvol.models.logmix import LogMix
from sdevpy.volatility.impliedvol.models import sabr
from sdevpy.volatility.impliedvol.numerical_impliedvol import NumericalImpliedVol
from sdevpy.volatility.localvol.localvol import ConstantLocalVol
from sdevpy.market import provider as mdp
from sdevpy.calibration import provider as cdp
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider
from sdevpy.volatility.impliedvol.models.cubicvol import (
    create_section, cubicvol_check_params, calculate_epsilon,
    is_valid_left, is_valid_right, sample_params
)
from scipy.stats import norm as scipy_norm
from sdevpy.volatility.impliedvol.models.logmix import (
    logmix_f, logmix_df, LogMixMean, LogMixVar, LogMixWeight, LogMixNorm,
    get_logmix_parameters
)
from sdevpy.maths import constants


# n_mix=1, flat vol term structure: beta=1, a=b=0.2, c=0, d=1 → stdev(t=1)=0.2
_LOGMIX_PARAMS_1 = [1.0, 0.2, 0.2, 0.0, 1.0]

_CALIB_NAME   = "ABC"
_CALIB_DATE   = dt.datetime(2025, 12, 15)
_CALIB_CONFIG = {'optimizer': 'SLSQP', 'tol': 1e-6}


def make_logmix1():
    """ Quick LogMix maker """
    m = LogMix(n_mix=1)
    m.update_params(_LOGMIX_PARAMS_1)
    return m


def test_cubicvol_create_section():
    config = {'atm': 0.25, 'skew': 0.1, 'kurt': 0.25, 'vl': 0.30, 'vr': 0.28}
    section = create_section(1.0, param_config=config)
    assert section.params[0] == 0.25

    section = create_section(1.0)
    d = section.dump_params()
    assert set(d.keys()) == {'atm', 'skew', 'kurt', 'vl', 'vr'}


def test_cubicvol_check_params():
    section = create_section(1.0)
    is_ok, penalty = section.check_params()
    assert is_ok
    assert penalty == 0.0

    section.params = np.array([-0.1, 0.1, 0.25, 0.30, 0.28])  # negative atm
    is_ok, penalty = section.check_params()
    assert not is_ok

    is_ok, penalty, epsl, epsr = cubicvol_check_params([0.25, 0.1])  # only 2 params
    assert not is_ok


def test_cubicvol_is_valid():
    # vl < atm → invalid
    assert not is_valid_left(atm=0.25, skew=0.1, kurt=0.25, vl=0.20)
    assert not is_valid_right(atm=0.25, skew=-0.1, kurt=0.25, vr=0.28)


def test_cubicvol_calculate_epsilon():
    # delta_v=0 and is_right=True, skew != 0 → returns kurt^2 / (4*skew)
    atm, skew, kurt, vr = 0.25, 0.1, 0.25, 0.25
    eps = calculate_epsilon(atm, skew, kurt, vr, is_right=True)
    expected = kurt * kurt / (4.0 * skew)
    assert abs(eps - expected) < 1e-10

    with pytest.raises(RuntimeError):
        calculate_epsilon(0.25, 0.1, 0.25, 0.20, is_right=False)  # vl < atm


####### LogMix ####################################################################################

def test_logmix_f():
    # t=1, beta=1: tmp=2, result = 1 - 2/5 = 0.6
    assert abs(logmix_f(1.0, 1.0) - 0.6) < 1e-12
    with pytest.raises(ValueError):
        logmix_f(1.0, 0.0)

    # t=1, beta=1: tmp1=2, tmp2=5, result = (4*2/1)/25 = 0.32
    assert abs(logmix_df(1.0, 1.0) - 0.32) < 1e-12
    with pytest.raises(ValueError):
        logmix_df(1.0, 0.0)


def test_logmix_mean():
    m = LogMixMean(mu0=0.5, beta=1.0)
    assert abs(m.value(1.0) - 0.5 * 0.6) < 1e-12  # 0.3
    assert abs(m.diff(1.0) - 0.5 * 0.32) < 1e-12  # 0.16


def test_logmix_var():
    # a=b=0.2, c=0, d=1: s=0.2 for all t → value(1) = 0.04
    v = LogMixVar(a=0.2, b=0.2, c=0.0, d=1.0)
    assert abs(v.value(1.0) - 0.04) < 1e-12

    # flat vol: ds=0, so diff = s*(0 + s) = s^2 = 0.04
    v = LogMixVar(a=0.2, b=0.2, c=0.0, d=1.0)
    assert abs(v.diff(1.0) - 0.04) < 1e-12

    # d=0 is floored to 1e-8; result must be finite and positive
    v = LogMixVar(a=0.2, b=0.2, c=0.0, d=0.0)
    assert np.isfinite(v.value(1.0)) and v.value(1.0) > 0.0


def test_logmix_norm():
    with pytest.raises(ValueError):
        LogMixNorm(w0=[0.6, 0.4], beta=[1.0])  # beta length mismatch

    # n_mix=1: norm = w0/logmix_f(1, 1) = 1.0/0.6
    n = LogMixNorm(w0=[1.0], beta=[1.0])
    assert abs(n.value(1.0) - 1.0 / 0.6) < 1e-12

    # n_mix=1: dnorm = -w0/f^2 * df = -1.0/0.36 * 0.32
    n = LogMixNorm(w0=[1.0], beta=[1.0])
    expected = -1.0 / (0.6 * 0.6) * 0.32
    assert abs(n.diff(1.0) - expected) < 1e-12


def test_logmix_weight():
    with pytest.raises(ValueError):
        LogMixWeight(component=0, w0=[0.6, 0.4], beta=[1.0])

    # n_mix=1: weight always equals 1.0
    wt = LogMixWeight(component=0, w0=[1.0], beta=[1.0])
    assert abs(wt.value(1.0) - 1.0) < 1e-12

    n = wt.norm.value(1.0)
    n_diff = wt.norm.diff(1.0)
    with pytest.raises(ValueError):
        wt.diff_given_norm(0.0, n, n_diff)  # t <= 0

    # For n_mix=1, weight=1 for all t, so diff=0
    wt = LogMixWeight(component=0, w0=[1.0], beta=[1.0])
    assert abs(wt.diff(1.0)) < 1e-12


def test_get_logmix_parameters_n_mix2():
    # n_mix=2, second component weight=0.3
    p = [0.2, 0.2, 0.2, 0.0, 1.0,   # component 0
         0.3, 0.1,                    # w1, shift1
         0.2, 0.2, 0.2, 0.0, 1.0]    # component 1
    d, is_ok = get_logmix_parameters(2, p)
    assert is_ok
    assert abs(d['w'][1] - 0.3) < 1e-12
    assert abs(d['w'][0] - 0.7) < 1e-12

def test_get_logmix_parameters_checks():
    with pytest.raises(ValueError):
        get_logmix_parameters(1, [0.2, 0.2])  # expects 5, got 2

    with pytest.raises(ValueError):
        get_logmix_parameters(0, [])

    # w1=1.5 → tmp_w = -0.5 < weight_floor → is_ok=False
    p = [0.2, 0.2, 0.2, 0.0, 1.0,
         1.5, 0.0,
         0.2, 0.2, 0.2, 0.0, 1.0]
    d, is_ok = get_logmix_parameters(2, p)
    assert not is_ok


def test_logmix_raises():
    m = LogMix(n_mix=1)
    with pytest.raises(RuntimeError):
        m.price(1.0, np.asarray([1.0]), True, 1.0)
    with pytest.raises(RuntimeError):
        m.pdf(1.0, np.asarray([1.0]), 1.0)
    with pytest.raises(ValueError):
        make_logmix1().pdf(0.0, np.asarray([1.0]), 1.0)
    with pytest.raises(RuntimeError):
        m.cdf(1.0, np.asarray([1.0]), 1.0)
    with pytest.raises(ValueError):
        make_logmix1().cdf(0.0, np.asarray([1.0]), 1.0)
    with pytest.raises(RuntimeError):
        m.check_params()
    with pytest.raises(RuntimeError):
        m.dump_data()


def test_logmix_n_params():
    assert LogMix(n_mix=1).n_params == 5
    assert LogMix(n_mix=2).n_params == 12
    assert LogMix(n_mix=3).n_params == 19
    assert len(LogMix(n_mix=1).initial_point()) == 5
    assert len(LogMix(n_mix=2).initial_point()) == 12
    b = LogMix(n_mix=1).bounds()
    assert len(b.lb) == 5
    b = LogMix(n_mix=2).bounds()
    assert len(b.lb) == 12
    m = make_logmix1()
    is_ok, penalty = m.check_params()
    assert is_ok
    assert penalty == 0.0
    m = LogMix(n_mix=2)
    # w1=1.5 → first weight negative → is_ok=False
    p = [0.2, 0.2, 0.2, 0.0, 1.0,  1.5, 0.0,  0.2, 0.2, 0.2, 0.0, 1.0]
    m.update_params(p)
    is_ok, penalty = m.check_params()
    assert not is_ok
    assert penalty == constants.FLOAT_INFTY
    m = LogMix(n_mix=1, check_fwd_var=True)
    m.update_params(_LOGMIX_PARAMS_1)  # flat 20% vol → fwd var always positive
    is_ok, penalty = m.check_params()
    assert is_ok
    m = make_logmix1()
    d = m.dump_data()
    assert d['type'] == 'LogMix1'
    assert len(d['params']) == 5


def test_logmix_black_call_atm_value():
    # ATM: f=k=1, stdev=0.2; Black ATM call = f*(2*N(stdev/2)-1)
    m = make_logmix1()
    test = m.black(1.0, True, 1.0, 0.2)
    # d1=0.1, d2=-0.1: call = N(0.1) - N(-0.1)
    expected = scipy_norm.cdf(0.1) - scipy_norm.cdf(-0.1)
    assert abs(test - expected) < 1e-12

    # ATM CDF for lognormal with mu=0: N(stdev/2) = N(0.1)
    test = m.cdf(1.0, np.asarray([1.0]), 1.0)
    expected = scipy_norm.cdf(0.1)
    assert abs(test - expected) < 1e-8


def test_logmix_taylor_parameters():
    m = make_logmix1()
    w, w_d, mu, mu_d, v, v_d = m.taylor_parameters(1.0)
    assert len(w) == 1 and len(v) == 1
    assert abs(w_d[0]) < 1e-12


def test_logmix_local_vol():
    # t < 5/365 → recursive call; result must still be positive
    m = make_logmix1()
    lv = m.local_vol(0.001, np.asarray([1.0]))
    assert np.all(lv >= 0.0)

    # ts=0 (< 5/365): uses te-based stdev; flat vol → result = 0.2
    lv = m.local_vol_step(0.0, 1.0, np.asarray([1.0]))
    assert abs(lv[0] - 0.2) < 1e-6

    # ts=0.5 >= 5/365; flat vol → result = 0.2
    m = make_logmix1()
    lv = m.local_vol_step(0.5, 1.0, np.asarray([1.0]))
    assert abs(lv[0] - 0.2) < 1e-6


def test_logmix_pdf():
    model = make_logmix1()
    expiry = 1.0
    fwd = 0.04
    strikes = np.asarray([0.03, 0.04, 0.05])
    test = model.pdf(expiry, strikes, fwd)
    ref = np.asarray([27.15024656, 49.61906844, 19.05342396])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_logmix_pdf_integrates_to_one():
    """Integral of pdf over (0, ∞) must be ≈ 1."""
    model = make_logmix1()
    integral, _ = quad(lambda k: model.pdf(1.0, k, 1.0), 0.2, 6.0)
    assert abs(integral - 1.0) < 1e-8


def test_logmix_cdf_consistent_with_pdf():
    """CDF(b) - CDF(a) must equal ∫_a^b pdf(k) dk"""
    model = make_logmix1()
    t, fwd, a, b = 1.0, 1.0, 0.8, 1.2

    cdf_diff = model.cdf(t, b, fwd) - model.cdf(t, a, fwd)
    integral, _ = quad(lambda k: model.pdf(t, k, fwd), a, b)
    assert abs(cdf_diff - integral) < 1e-5


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
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_logmix_from_file():
    name, date = 'ABC', dt.datetime(2025, 12, 15)
    cal_prov = CalibrationDataFileProvider()
    ivol = cdp.get_impliedvol(name, date, 'LogMix3', cal_prov)
    n_mix = ivol.n_mix
    params = ivol.params
    assert n_mix == 3
    assert len(params) == 19


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
    assert isequal(test, 0.04752866204709)
    # assert isequal(test, 0.065354183239)


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
    ref = np.asarray([0.28002666, 0.27620326, 0.27544121])
    # ref = np.asarray([0.41214181, 0.23553742, 0.20534387])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


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
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


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
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_biexp_formula():
    t = 0.5
    params = biexp.sample_params(t)
    m = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(m) # Log-moneyness

    test = biexp.biexp_formula(t, log_m, params)
    ref = np.asarray([0.29982632, 0.25, 0.27999992])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_cubicvol_formula():
    t = 1.5
    atm, skew, kurt, vl, vr = 0.25, 0.1, 0.25, 0.30, 0.27
    m = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(m) # Log-moneyness

    test = cubicvol.cubicvol(t, log_m, atm, skew, kurt, vl, vr)
    ref = np.asarray([0.3, 0.25, 0.249886])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_vsvi_formula():
    t = 1.5
    vstar, b, rho, xstar, lambda_ = 0.25, 0.1 / t, -0.25, 0.0, 0.25 * t
    mx = np.asarray([0.5, 1.0, 2.0]) # Moneyness
    log_m = np.log(mx) # Log-moneyness

    test = vsvi.vsvi(log_m, vstar, b, rho, xstar, lambda_)
    # print(test)
    ref = np.asarray([0.28327209, 0.25, 0.27122271])
    # ref = np.asarray([0.31409145, 0.275, 0.29098655])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_sabr():
    # Test near ATM
    expiry = 0.5
    fwd = 0.04
    params = {'LnVol': 0.25, 'Beta': 0.4, 'Nu': 0.50, 'Rho': -0.25}
    strikes = np.asarray([0.01, 0.04, 0.06])
    test = sabr.sabr_from_dict(expiry, strikes, fwd, params)
    ref = np.asarray([0.54225604, 0.25208659, 0.22711695])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


########### NumericalImpliedVol ###################################################################
LV = ConstantLocalVol(0.20, valdate=dt.datetime(2025, 12, 15))


def test_numerical_impliedvol_dump_data_structure():
    niv = NumericalImpliedVol(LV)
    data = niv.dump_data()
    assert 'lv' in data


def test_numerical_impliedvol_calculate_prices_returns_array():
    niv = NumericalImpliedVol(LV)
    fwd = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    prices = niv.calculate_prices(1.0, strikes, 'call', fwd)
    assert len(prices) == 3
    assert np.all(prices >= 0.0)


def test_numerical_impliedvol_calculate_call_put_parity():
    # C - P = F - K for each strike (forward premium, no discounting)
    niv = NumericalImpliedVol(LV)
    fwd, t = 100.0, 5.0
    strikes = np.array([95.0, 100.0, 105.0])
    calls = niv.calculate_prices(t, strikes, 'call', fwd)
    puts = niv.calculate_prices(t, strikes, 'put', fwd)
    for i, k in enumerate(strikes):
        # print(calls[i] - puts[i])
        # print(fwd - k)
        assert np.isclose(calls[i] - puts[i], fwd - k, atol=0.2)


def test_tssvi1_calibrate():
    """ Full round-trip: load market data → calibrate → check RMSE and validity """
    md = MarketDataFileProvider()
    fwd_curve = mdp.get_eq_forward_curves([_CALIB_NAME], _CALIB_DATE, md)[0]
    option_data = md.get_eq_vol_data(_CALIB_NAME, _CALIB_DATE)
    mkt_data = {'option_data': option_data, 'forward_curve': fwd_curve}

    model = TsSvi1()
    calibrator = TsIvCalibrator(model, _CALIB_CONFIG)
    calibrator.calibrate(mkt_data)

    # Model must be set at the optimum after calibrate()
    assert model.params is not None

    # Cross-strike RMSE must be under 50 bps
    assert calibrator.result.fun < 0.005

    # All model vols must be finite and positive at the calibrated params
    vols = model.calculate(calibrator.times, calibrator.strikes, True, calibrator.fwds)
    assert np.all(np.isfinite(vols))
    assert np.all(vols > 0)


if __name__ == "__main__":
    print("Hello")
    test_logmix_from_file()
    # test_numerical_impliedvol_calculate_call_put_parity()
    # test_build_step_grid_short_term()
    # test_logmix_from_file()
    # test_logmix_pdf_integrates_to_one()
    # test_logmix_pdf()
    # test_sabr()
    # test_tssvi1_objective()
    # test_tssvi1()
    # test_svi_formula()
    # test_biexp_formula()
    # test_cubicvol_formula()
    # test_vsvi_formula()
