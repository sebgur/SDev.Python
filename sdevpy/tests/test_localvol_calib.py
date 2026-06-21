import pytest
from unittest.mock import MagicMock
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
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider
from sdevpy.volatility.localvol.dupire_calib import dupire_formula_single   # add to existing import
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol, MatrixLocalVol  # add MatrixLocalVol
from sdevpy.volatility.localvol.lvsection_calib import calibrate_lv_bysections, LvObjectiveBuilder, PenaltyType
from sdevpy.volatility.impliedvol.impliedvol import LvMethod
from sdevpy.analytics import black
from sdevpy.maths import metrics, constants    # add constants to existing


CALIB_VALDATE = dt.datetime(2025, 12, 15)
CALIB_NAME = "ABC"
CALIB_CONFIG = {'model_name': 'BiExp', 'optimizer': 'SLSQP',
                'pde_timesteps': 10, 'pde_spotsteps': 30, 'sol_as_init': False}


##################### Other paths ##############################################################
##################### Test helpers for new tests ##################################################

def _mock_ivsurf(lv_method, black_vol=0.20, dvar_dt=0.04, dvol_dx=0.0, d2vol_dx2=0.0, density=None):
    m = MagicMock()
    m.lv_method = lv_method
    m.black_volatility.return_value = black_vol
    m.dvariance_dt.return_value = dvar_dt
    m.dvolatility_dx.return_value = dvol_dx
    m.d2volatility_dx2.return_value = d2vol_dx2
    if density is not None:
        m.density.return_value = density
    return m


def _make_builder(lv_t_grid=None, expiry_grid=None, option_type='straddle',
                  penalty_type=PenaltyType.INFINITY):
    """Create an LvObjectiveBuilder backed by a minimal mock LV that passes all validation."""
    if expiry_grid is None:
        expiry_grid = [1.0, 2.0]
    if lv_t_grid is None:
        lv_t_grid = [0.0, expiry_grid[0]]
    mock_lv = MagicMock()
    mock_lv.t_grid = lv_t_grid
    n = len(expiry_grid)
    cf_prices = [np.array([0.005, 0.004])] * n
    return LvObjectiveBuilder(mock_lv, expiry_grid, [0.04] * n, [[0.03, 0.04]] * n, cf_prices,
                              MagicMock(), option_type=option_type, penalty_type=penalty_type)


##################### dupire_formula_single #######################################################

class TestDupireFormulaSingle:
    def test_ts_near_zero_returns_black_vol_at_te(self):
        """ts < t_threshold: early return using black_volatility(te, x, 1.0)"""
        surf = _mock_ivsurf(LvMethod.ImpliedVol, black_vol=0.25)
        result = dupire_formula_single(surf, ts=0.0, te=1.0, x=1.0)
        assert result == pytest.approx(0.25)
        surf.black_volatility.assert_called_with(1.0, 1.0, 1.0)

    def test_x_near_zero_returns_sqrt_dvar_dt(self):
        """x < x_threshold: returns sqrt of forward variance derivative"""
        surf = _mock_ivsurf(LvMethod.ImpliedVol, dvar_dt=0.04)
        result = dupire_formula_single(surf, ts=0.5, te=1.0, x=1e-7)
        assert result == pytest.approx(np.sqrt(0.04))

    def test_implied_vol_zero_returns_zero(self):
        """ImpliedVol branch, theta == 0: returns 0"""
        surf = _mock_ivsurf(LvMethod.ImpliedVol, black_vol=0.0, dvar_dt=0.04)
        result = dupire_formula_single(surf, ts=0.5, te=1.0, x=1.0)
        assert result == pytest.approx(0.0)

    def test_implied_vol_negative_sigma2_clamps_to_zero(self):
        """ImpliedVol branch, very large negative curvature → denominator < 0 → sigma2 < 0 → 0"""
        # With dvol_dx=0 and d2vol_dx2=-100 at x=1, ts=1: denominator = 0.20*1*(-100) + 1 = -19 < 0
        surf = _mock_ivsurf(LvMethod.ImpliedVol, black_vol=0.20, dvar_dt=0.04,
                            dvol_dx=0.0, d2vol_dx2=-100.0)
        result = dupire_formula_single(surf, ts=1.0, te=2.0, x=1.0)
        assert result == pytest.approx(0.0)

    def test_implied_vol_method_matches_vectorised(self):
        """scalar dupire_formula_single must agree with the vectorised dupire_formula at the same point"""
        surf = make_tssvi1()
        scalar = dupire_formula_single(surf, ts=0.25, te=1.0, x=1.0)
        vector = dupire_formula(surf, ts=0.25, te=1.0, x=np.array([1.0]))[0]
        assert scalar == pytest.approx(vector, rel=1e-6)

    def test_pdf_stddev_zero_returns_zero(self):
        """PDF branch, theta == 0 → stddev == 0: returns 0"""
        surf = _mock_ivsurf(LvMethod.PDF, black_vol=0.0, dvar_dt=0.04)
        result = dupire_formula_single(surf, ts=0.5, te=1.0, x=1.0)
        assert result == pytest.approx(0.0)

    def test_pdf_method_matches_vectorised(self):
        """scalar dupire_formula_single (PDF path) must agree with the vectorised version"""
        surf = make_logmix2()
        scalar = dupire_formula_single(surf, ts=0.25, te=1.0, x=1.0)
        vector = dupire_formula(surf, ts=0.25, te=1.0, x=np.array([1.0]))[0]
        assert scalar == pytest.approx(vector, rel=1e-4)

    def test_invalid_method_raises(self):
        surf = _mock_ivsurf("bad_method", dvar_dt=0.04)
        with pytest.raises(ValueError, match="Invalid Dupire"):
            dupire_formula_single(surf, ts=0.5, te=1.0, x=1.0)


##################### dupire_formula: additional uncovered paths ##################################

class TestDupireFormulaAdditional:
    def test_analytical_method_delegates_to_local_vol_step(self):
        """LvMethod.Analytical: delegates immediately to ivsurf.local_vol_step"""
        surf = MagicMock()
        surf.lv_method = LvMethod.Analytical
        surf.local_vol_step.return_value = np.array([0.25, 0.22])
        x = np.array([0.9, 1.1])
        result = dupire_formula(surf, ts=0.5, te=1.0, x=x)
        surf.local_vol_step.assert_called_once()  # called exactly once before returning
        np.testing.assert_array_equal(result, np.array([0.25, 0.22]))

    def test_invalid_method_raises(self):
        """Unknown lv_method must raise ValueError after dvariance_dt is computed"""
        surf = MagicMock()
        surf.lv_method = "bad_method"
        surf.dvariance_dt.return_value = np.array([0.04])
        with pytest.raises(ValueError, match="Invalid Dupire"):
            dupire_formula(surf, ts=0.5, te=1.0, x=np.array([1.0]))

    def test_near_zero_moneyness_overrides_sigma2_with_dvar_dt(self):
        """x[0] << x_threshold: sigma2 overridden with dvar_dt → result = sqrt(dvar_dt)"""
        surf = MagicMock()
        surf.lv_method = LvMethod.ImpliedVol
        surf.dvariance_dt.return_value = np.array([0.04, 0.04])
        surf.taylor_dx.return_value = (np.array([0.20, 0.20]),
                                       np.array([0.0, 0.0]),
                                       np.array([0.0, 0.0]))
        x = np.array([1e-7, 1.0])
        result = dupire_formula(surf, ts=0.5, te=1.0, x=x)
        assert result[0] == pytest.approx(np.sqrt(0.04))


##################### calib_lv_dupire: additional uncovered paths ################################

class TestCalibLvDupireAdditional:
    def test_returns_matrix_local_vol(self):
        result = calib_lv_dupire(make_tssvi1(), points_per_year=2, n_strikes=3)
        assert isinstance(result['lv'], MatrixLocalVol)

    def test_last_slice_copies_second_to_last(self):
        """moneynesses[-1] and lv[-1] must be identical objects to [-2] (line 182)"""
        result = calib_lv_dupire(make_tssvi1(), points_per_year=2, n_strikes=3)
        assert result['lv_matrix'][-1] is result['lv_matrix'][-2]
        assert result['moneyness'][-1] is result['moneyness'][-2]

    def test_explicit_t_grid_is_used_verbatim(self):
        """When t_grid is passed explicitly the grid builder is skipped"""
        t_grid = np.array([0.0, 0.5, 1.0])
        result = calib_lv_dupire(make_tssvi1(), t_grid=t_grid, n_strikes=3)
        np.testing.assert_array_equal(result['t_grid'], t_grid)

    def test_grid_in_percents_produces_different_moneyness(self):
        """grid_in_percents=True spaces strikes in quantile space, not linearly"""
        result_std = calib_lv_dupire(make_tssvi1(), points_per_year=2, n_strikes=5, grid_in_percents=False)
        result_pct = calib_lv_dupire(make_tssvi1(), points_per_year=2, n_strikes=5, grid_in_percents=True)
        assert len(result_pct['moneyness'][0]) == 5
        assert not np.allclose(result_pct['moneyness'][0], result_std['moneyness'][0], rtol=0, atol=1e-6)


##################### LvObjectiveBuilder.__init__: validation errors ##############################

class TestLvObjectiveBuilderInitValidation:
    def _mock_lv(self, t_grid):
        m = MagicMock()
        m.t_grid = t_grid
        return m

    def test_raises_when_t_grid_has_single_point(self):
        mock_lv = self._mock_lv([0.0])
        with pytest.raises(ValueError, match="only has 1 point"):
            LvObjectiveBuilder(mock_lv, [1.0], [0.04], [[0.04]], [np.array([0.005])], MagicMock())

    def test_raises_on_size_mismatch(self):
        mock_lv = self._mock_lv([0.0, 1.0, 2.0])   # len=3, expiry_grid len=2
        with pytest.raises(ValueError, match="Inconsistent sizes"):
            LvObjectiveBuilder(mock_lv, [1.0, 2.0], [0.04]*2, [[0.04]]*2,
                               [np.array([0.005])]*2, MagicMock())

    def test_raises_when_t_grid_does_not_start_at_zero(self):
        mock_lv = self._mock_lv([0.1, 1.0])         # starts at 0.1, not 0.0
        with pytest.raises(ValueError, match="does not start at 0"):
            LvObjectiveBuilder(mock_lv, [1.0, 2.0], [0.04]*2, [[0.04]]*2,
                               [np.array([0.005])]*2, MagicMock())

    def test_raises_on_inconsistent_time_values(self):
        mock_lv = self._mock_lv([0.0, 0.5])         # lv.t_grid[1]=0.5 != expiry_grid[0]=1.0
        with pytest.raises(ValueError, match="Inconsistent time values"):
            LvObjectiveBuilder(mock_lv, [1.0, 2.0], [0.04]*2, [[0.04]]*2,
                               [np.array([0.005])]*2, MagicMock())


##################### LvObjectiveBuilder.objective: penalty paths ################################

class TestLvObjectiveBuilderPenalties:
    def _make_rejected_builder(self, penalty_type):
        builder = _make_builder(penalty_type=penalty_type)
        builder.exp_idx = 0
        builder.cf_prices = np.array([0.005, 0.004])
        builder.lv.check_params.return_value = (False, 0.5)   # mock: params rejected, model penalty=0.5
        return builder

    def test_model_penalty_returns_model_value(self):
        builder = self._make_rejected_builder(PenaltyType.MODEL)
        assert builder.objective(np.array([0.1, 0.2])) == pytest.approx(0.5)

    def test_prices_penalty_returns_cf_price_sum(self):
        builder = self._make_rejected_builder(PenaltyType.PRICES)
        assert builder.objective(np.array([0.1, 0.2])) == pytest.approx(0.009)

    def test_infinity_penalty_returns_float_infty(self):
        builder = self._make_rejected_builder(PenaltyType.INFINITY)
        assert builder.objective(np.array([0.1, 0.2])) == constants.FLOAT_INFTY


##################### LvObjectiveBuilder.residuals: rejected path ################################

def test_lv_objective_builder_residuals_rejected():
    """When params are rejected, residuals() returns a constant vector of sqrt(sum(cf_prices))"""
    builder = _make_builder()
    cf_prices = np.array([0.005, 0.004])
    builder.exp_idx = 0
    builder.cf_prices = cf_prices
    builder.lv.check_params.return_value = (False, 0.0)
    result = builder.residuals(np.array([0.1, 0.2]))
    expected = np.full(len(cf_prices), np.sqrt(cf_prices.sum()))
    np.testing.assert_array_almost_equal(result, expected)


##################### LvObjectiveBuilder.calculate_vols: CALL, PUT, invalid ######################

class TestLvObjectiveBuilderCalculateVols:
    _T, _K, _F, _VOL = 1.0, 0.04, 0.04, 0.20

    def _make_vol_builder(self, option_type):
        builder = _make_builder(option_type=option_type)
        builder.exp_idx = 0
        builder.expiry_grid = [self._T, 2.0]
        builder.fwd = self._F
        builder.strikes = [self._K]
        return builder

    def test_call_option_type_round_trips_call_price(self):
        builder = self._make_vol_builder('call')
        call_price = float(np.asarray(black.price(self._T, self._K, True, self._F, self._VOL)).flat[0])
        builder.pde_prices = [call_price]
        vols = builder.calculate_vols()
        assert vols[0] == pytest.approx(self._VOL, rel=1e-4)

    def test_put_option_type_round_trips_put_price(self):
        builder = self._make_vol_builder('put')
        put_price = float(np.asarray(black.price(self._T, self._K, False, self._F, self._VOL)).flat[0])
        builder.pde_prices = [put_price]
        vols = builder.calculate_vols()
        assert vols[0] == pytest.approx(self._VOL, rel=1e-4)

    def test_unknown_option_type_raises(self):
        builder = self._make_vol_builder('straddle')
        builder.option_type = "bad"    # bypass string_to_optiontype; no OptionType enum member matches
        builder.pde_prices = [0.005]
        with pytest.raises(ValueError, match="Unknown option type"):
            builder.calculate_vols()


##################### calibrate_lv_bysections: penalty_type guard ################################

def test_calibrate_lv_bysections_unknown_penalty_type_raises():
    """'bogus' penalty_type must raise before any calibration loop starts"""
    config = {'model_name': 'BiExp', 'pde_timesteps': 10, 'pde_spotsteps': 30,
              'force_restart': True, 'penalty_type': 'bogus'}
    with pytest.raises(ValueError, match="Unsupported penalty type"):
        calibrate_lv_bysections(CALIB_VALDATE, CALIB_NAME, config,
                                MarketDataFileProvider(), CalibrationDataFileProvider())


##################### End Other paths ##############################################################

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
    assert np.allclose(m_test, m_ref, rtol=0.0, atol=1e-8)
    lv_test = np.asarray(lv[0])
    # print(lv_test)
    lv_ref = np.asarray([0.3291237, 0.16782647, 0.16614304])
    # lv_ref = np.asarray([0.33384114, 0.18219919, 0.36166036])
    assert np.allclose(lv_test, lv_ref, rtol=0.0, atol=1e-8)


def test_dupire_impliedvol():
    """ Check Dupire formula by implied vol method """
    x = np.asarray([0.9, 1.0, 1.1])
    test = dupire_formula(make_tssvi1(), ts=0.25, te=1.0, x=x)
    # print(test)
    ref = np.asarray([0.37100141, 0.24239379, 0.20644008])
    # ref = np.asarray([0.36949807, 0.2413907, 0.20621366])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_dupire_pdf():
    """ Check Dupire formula by PDF method """
    x = np.asarray([0.9, 1.0, 1.1])
    test = dupire_formula(make_logmix2(), ts=0.25, te=1.0, x=x)
    ref = np.asarray([0.20, 0.20, 0.20])
    # ref = np.asarray([0.20000181, 0.20001192, 0.1999994])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


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
    assert np.allclose(lv, expected, rtol=0.0, atol=1e-8)


def test_dupire_flat_surface_recovers_constant_vol():
    """ On a flat vol surface (no skew, flat term structure) Dupire LV = IV """
    x = np.asarray([0.8, 0.9, 1.0, 1.1, 1.2])
    lv = dupire_formula(make_flat_surface(), ts=0.5, te=1.0, x=x)
    assert np.allclose(lv, FLAT_VOL, rtol=0.0, atol=1e-8)


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

    assert np.allclose(lv.t_grid, calib_times, rtol=0.0, atol=1e-8)
    assert np.allclose(lv.vol_grid, calib_lvols, rtol=0.0, atol=1e-8)


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

    assert isequal(lv.vol, calib_lvol, 1e-8)


##################### Calib LvSection #############################################################

def test_calibrate_lv_bysections():
    """ Calibrated LV must reproduce market vols within 200 bps RMSE at each expiry """
    calib_config = CALIB_CONFIG.copy()
    calib_config['force_restart'] = True
    md_prov = MarketDataFileProvider()
    cal_prov = CalibrationDataFileProvider()
    result = calibrate_lv_bysections(CALIB_VALDATE, CALIB_NAME, calib_config, md_prov, cal_prov,
                                     calc_pde_vols=True)
    lv, iv_data, pde_vols = result['lv'], result['iv_data'], result['pde_vols']

    # Check output consistency
    n_expiries = len(iv_data.expiries) # 6 for ABC 2025-12-15
    assert isinstance(lv, InterpolatedParamLocalVol)
    assert len(lv.t_grid) == n_expiries
    assert lv.name == CALIB_NAME
    assert lv.valdate == CALIB_VALDATE

    # Check accuracy
    for mkt_vols, exp_pde_vols in zip(iv_data.vols, pde_vols, strict=True):
        assert all(v > 0.0 for v in exp_pde_vols)
        assert metrics.rmse(mkt_vols, exp_pde_vols) < 0.02


def test_calibrate_lv_bysections_least_squares():
    """ Calibrated LV must reproduce market vols within 200 bps RMSE at each expiry """
    calib_config = CALIB_CONFIG.copy()
    calib_config['optimizer'] = 'LeastSquares'
    calib_config['model_name'] = "VSVI"
    md_prov = MarketDataFileProvider()
    cal_prov = CalibrationDataFileProvider()
    result = calibrate_lv_bysections(CALIB_VALDATE, CALIB_NAME, calib_config, md_prov, cal_prov,
                                     calc_pde_vols=True)
    lv, iv_data, pde_vols = result['lv'], result['iv_data'], result['pde_vols']
    # iv_data, pde_vols = result['iv_data'], result['pde_vols']

    # Check output consistency
    n_expiries = len(iv_data.expiries) # 6 for ABC 2025-12-15
    assert isinstance(lv, InterpolatedParamLocalVol)
    assert len(lv.t_grid) == n_expiries
    assert lv.name == CALIB_NAME
    assert lv.valdate == CALIB_VALDATE

    # Check accuracy
    for mkt_vols, exp_pde_vols in zip(iv_data.vols, pde_vols, strict=True):
        assert all(v > 0.0 for v in exp_pde_vols)
        # print(metrics.rmse(mkt_vols, exp_pde_vols))
        assert metrics.rmse(mkt_vols, exp_pde_vols) < 0.025

if __name__ == "__main__":
    print("Hello")
    test_calibrate_lv_bysections()
    # test_calibrate_lv_bysections_least_squares()
    # test_dupire_impliedvol()
