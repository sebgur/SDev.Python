import numpy as np
import datetime as dt
from sdevpy.volatility.localvol.dupire_calib import dupire_formula, calib_lv_dupire
from sdevpy.volatility.localvol.black_calib import calib_lv_black
from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
from sdevpy.tests.test_localvol import (make_tssvi1, make_tssvi2, make_logmix2)
from sdevpy.tests.test_localvol import (make_flat_surface, FLAT_VOL)
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol
from sdevpy.volatility.localvol.lvsection_calib import calibrate_lv_bysections
from sdevpy.maths import metrics
from sdevpy.utilities import timegrids
from sdevpy.utilities.tools import isequal


CALIB_VALDATE = dt.datetime(2025, 12, 15)
CALIB_NAME = "ABC"
CALIB_CONFIG = {'start_new': True, 'model': 'BiExp', 'optimizer': 'SLSQP',
                'tol': 1e-6, 'pde_timesteps': 10, 'pde_spotsteps': 30, 'sol_as_init': False}



##################### Dupire formula ##############################################################
def test_calib_dupire():
    """ Calibrate LV with Dupire """
    # Set IV surface model
    surface = TsSvi1()
    surface.update_params(surface.initial_point())

    # Calibrate
    result = calib_lv_dupire(surface, points_per_year=4, n_strikes=3)
    lv = result['lv_matrix']
    m = result['moneyness']
    t = result['t_grid']
    assert (np.abs(t[1] - 0.2857142857142857) < 1e-10)
    m_test = np.asarray(m[0])
    # print(m_test)
    m_ref = np.asarray([0.79519045, 1.020546, 1.24590155])
    assert np.allclose(m_test, m_ref, 1e-10)
    lv_test = np.asarray(lv[0])
    # print(lv_test)
    lv_ref = np.asarray([0.3291237, 0.16782647, 0.16614304])
    # lv_ref = np.asarray([0.33384114, 0.18219919, 0.36166036])
    assert np.allclose(lv_test, lv_ref, 1e-10)


def test_dupire_impliedvol():
    """ Check Dupire formula by implied vol method """
    x = np.asarray([0.9, 1.0, 1.1])
    test = dupire_formula(make_tssvi1(), ts=0.25, te=1.0, x=x)
    ref = np.asarray([0.36949807, 0.2413907, 0.20621366])
    assert np.allclose(test, ref, 1e-10)


def test_dupire_pdf():
    """ Check Dupire formula by PDF method """
    x = np.asarray([0.9, 1.0, 1.1])
    test = dupire_formula(make_logmix2(), ts=0.25, te=1.0, x=x)
    ref = np.asarray([0.20, 0.20, 0.20])
    # ref = np.asarray([0.20000181, 0.20001192, 0.1999994])
    assert np.allclose(test, ref, 1e-10)


def test_dupire_output_shape():
    """ Output array shape must match input x shape """
    x = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])
    lv = dupire_formula(make_tssvi2(), ts=0.25, te=1.0, x=x)
    assert lv.shape == x.shape


def test_dupire_scalar_input():
    """ Scalar x must return a scalar (0-d array) """
    lv = dupire_formula(make_tssvi2(), ts=0.25, te=1.0, x=1.0)
    assert np.ndim(lv) == 0


def test_dupire_ts_near_zero_returns_spot_vol():
    """ When ts < t_threshold the formula falls back to black_volatility(te, x) """
    surface = make_tssvi2()
    x = np.asarray([0.9, 1.0, 1.1])
    lv = dupire_formula(surface, ts=0.0, te=1.0, x=x)
    expected = surface.black_volatility(t=1.0, k=x, f=1.0)
    assert np.allclose(lv, expected)


def test_dupire_flat_surface_recovers_constant_vol():
    """ On a flat vol surface (no skew, flat term structure) Dupire LV = IV """
    x = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])
    lv = dupire_formula(make_flat_surface(), ts=0.5, te=1.0, x=x)
    assert np.allclose(lv, FLAT_VOL, atol=1e-6)


##################### LV calib (Black case) #######################################################
def test_lv_calib_black():
    # Pick ivol
    iv_surface = make_tssvi1()
    valdate = dt.datetime(2025, 12, 15)
    iv_surface.base_date = valdate

    # Calibrate Black LV through main function
    dates = np.asarray([dt.datetime(2026, 3, 15), dt.datetime(2026, 9, 15), dt.datetime(2027, 12, 15)])
    fwds = np.full(len(dates), 1.0) # Only caring about ATM
    strikes = fwds # Only caring about ATM
    lv_result = calib_lv_black(iv_surface, valdate, dates, strikes, fwds)
    lv = lv_result['lv']

    # Calibrate by hand
    times = timegrids.model_time(valdate, dates)
    calib_times, calib_lvols = [], []
    for i in range(len(dates)):
        te = times[i]
        ke = strikes[i]
        fe = fwds[i]
        ivole = iv_surface.black_volatility(te, ke, fe)
        if i == 0:
            ts, ivols = 0.0, 0.0
        else:
            ts = times[i - 1]
            ks = strikes[i - 1]
            fs = fwds[i - 1]
            ivols = iv_surface.black_volatility(ts, ks, fs)

        calib_times.append(ts)
        fwd_vol2 = (ivole**2 * te - ivols**2 * ts) / (te - ts)
        calib_lvols.append(np.sqrt(np.maximum(fwd_vol2, 0.0)))

    # # Create sampling grid
    # print(lv.t_grid)
    # print(lv.vol_grid)

    # sample_t = [0.0]
    # sample_t.append(calib_times[0] / 2.0)
    # sample_t.append(calib_times[0])
    # sample_t.append((calib_times[1] + calib_times[0]) / 2.0)
    # sample_t.append(calib_times[1])
    # sample_t.append((calib_times[2] + calib_times[1]) / 2.0)
    # sample_t.append(calib_times[2])
    # sample_t.append(calib_times[2] + 1.0)

    # for t in sample_t:
    #     print(f"{t}: {lv.value(t, [0.0])}")

    assert np.allclose(lv.t_grid, calib_times, 1e-10)
    assert np.allclose(lv.vol_grid, calib_lvols, 1e-10)


def test_lv_calib_black_constant():
    # Pick ivol
    iv_surface = make_tssvi1()
    valdate = dt.datetime(2025, 12, 15)
    iv_surface.base_date = valdate

    # Calibrate Black LV through main function
    dates = np.asarray([dt.datetime(2027, 12, 15)])
    fwds = np.full(len(dates), 1.0) # Only caring about ATM
    strikes = fwds # Only caring about ATM
    lv_result = calib_lv_black(iv_surface, valdate, dates, strikes, fwds)
    lv = lv_result['lv']

    # Calibrate by hand
    times = timegrids.model_time(valdate, dates)
    te, ke, fe = times[0], strikes[0], fwds[0]
    calib_lvol = iv_surface.black_volatility(te, ke, fe)

    assert isequal(lv.vol, calib_lvol, 1e-10)


##################### Calib LvSection #############################################################
def test_calibrate_lv_bysections_output_structure():
    """ Output must have lv with correct section count, iv_data matching the vol surface, empty pde_vols """
    result = calibrate_lv_bysections(CALIB_VALDATE, CALIB_NAME, CALIB_CONFIG)
    lv, iv_data, pde_vols = result['lv'], result['iv_data'], result['pde_vols']

    n_expiries = len(iv_data.expiries)  # 6 for ABC 2025-12-15
    assert isinstance(lv, InterpolatedParamLocalVol)
    assert len(lv.t_grid) == n_expiries
    assert lv.name == CALIB_NAME
    assert lv.valdate == CALIB_VALDATE
    assert pde_vols == []


def test_calibrate_lv_bysections_fit_quality():
    """ Calibrated LV must reproduce market vols within 200 bps RMSE at each expiry """
    result = calibrate_lv_bysections(CALIB_VALDATE, CALIB_NAME, CALIB_CONFIG, calc_pde_vols=True)
    iv_data, pde_vols = result['iv_data'], result['pde_vols']

    for mkt_vols, exp_pde_vols in zip(iv_data.vols, pde_vols, strict=True):
        assert all(v > 0.0 for v in exp_pde_vols)
        assert metrics.rmse(mkt_vols, exp_pde_vols) < 0.02


if __name__ == "__main__":
    print("Hello")
    test_calibrate_lv_bysections_output_structure()
