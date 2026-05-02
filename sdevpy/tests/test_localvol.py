import numpy as np
import numpy.typing as npt
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol, MatrixLocalVol
from sdevpy.volatility.impliedvol.models.svi import SviSection
from sdevpy.volatility.localvol.dupire import dupire_formula, calib_lv_dupire
from sdevpy.volatility.impliedvol.models.tssvi1 import TsSvi1
from sdevpy.volatility.impliedvol.models.tssvi2 import TsSvi2
from sdevpy.volatility.impliedvol.models.logmix import LogMix
from sdevpy.volatility.localvol.localvol_calib import LvObjectiveBuilder
from sdevpy.pde import forwardpde as fpde
from sdevpy.analytics import black


############### TEST HELPERS ######################################################################
VALID_PARAMS = np.array([0.04, 0.1, 0.0, 0.0, 0.2])  # a, b, rho, m, sigma
T_GRID    = np.array([0.25, 0.5, 1.0, 2.0])
LOGM_GRID = np.array([-0.25, -0.2, 0.0, 0.2, 0.25])
FLAT_VOL = 0.20

# v0, vinf, b_, tau, alpha, beta, r, x0star, lambda0, gamma, delta
FLAT_TSSVI1_PARAMS = [FLAT_VOL, FLAT_VOL, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.1, 1.0, 1.0]

_T_GRID  = np.array([0.5, 1.0])
FWD     = 100.0
FWDS    = [FWD, FWD]
STRIKES = [np.array([90.0, 100.0, 110.0]),
           np.array([90.0, 100.0, 110.0])]
# REF_VOL = 0.20

# a=0.04, b=0.001>0, rho=0, m=0, sigma=0.1>0 — all svi_check_params constraints satisfied
VALID_SVI   = np.array([0.04, 0.001, 0.0, 0.0, 0.10])
# b < 0 — fails svi_check_params immediately
INVALID_SVI = np.array([0.04, -0.10, 0.0, 0.0, 0.10])

def make_lv_by_sections(t_grid: list[float]=None, params: npt.ArrayLike=None):
    if t_grid is None:
        t_grid = [0.25, 0.5, 1.0, 2.0]
    if params is None:
        params = VALID_PARAMS
    sections = [SviSection(t) for t in t_grid]
    lv = InterpolatedParamLocalVol(sections)
    for i in range(len(t_grid)):
        lv.update_params(i, params)
    return lv

def make_vol_matrix(t_grid=T_GRID, logm_grid=LOGM_GRID):
    """ Bilinear test surface: vol = 0.20 + 0.02*t - 0.05*logm.
        Linear interpolation reproduces this exactly at any interior point. """
    t, lm = np.meshgrid(t_grid, logm_grid, indexing='ij')
    return lv_func_def(t, lm) # 0.20 + 0.02 * t - 0.05 * lm

def lv_func_def(t, lm):
    return 0.20 + 0.02 * t - 0.05 * lm

def make_mlv(**kwargs):
    """ Make MatrixLocalVol """
    return MatrixLocalVol(T_GRID, LOGM_GRID, make_vol_matrix(), **kwargs)

def make_flat_surface():
    s = TsSvi1()
    s.update_params(FLAT_TSSVI1_PARAMS)
    return s

def make_tssvi1():
    s = TsSvi1()
    s.update_params(s.initial_point())
    return s

def make_tssvi2():
    s = TsSvi2()
    s.update_params(s.initial_point())
    return s

def make_logmix2():
    s = LogMix(n_mix=2)
    s.update_params(s.initial_point())
    return s

def make_iplv(params=VALID_SVI):
    """ Make InterpolatedParamLocalVol """
    sections = [SviSection(t) for t in _T_GRID]
    lv = InterpolatedParamLocalVol(sections)
    for i in range(len(_T_GRID)):
        lv.update_params(i, params)
    return lv

def make_prices():
    return [black.price(exp, STRIKES[i], True, FWD, FLAT_VOL)
            for i, exp in enumerate(_T_GRID)]

def make_pde_config():
    return fpde.PdeConfig(n_timesteps=5, n_meshes=30, mesh_vol=FLAT_VOL,
                          scheme='rannacher', rescale_x=True, rescale_p=True)

def make_builder():
    return LvObjectiveBuilder(make_iplv(), FWDS, STRIKES, make_prices(), make_pde_config())


##################### LV calib by sections ########################################################
def test_lv_bysections_builder_initialize_density_integrates_to_one():
    """ Initial lognormal density on the log-spot grid must integrate to ≈ 1 """
    builder = make_builder()
    old_x, _, old_p = builder.initialize()
    assert abs(np.trapezoid(old_p, old_x) - 1.0) < 0.00001


def test_lv_bysections_builder_set_expiry_updates_slice_state():
    builder = make_builder()
    old_x, old_dx, old_p = builder.initialize()
    builder.set_expiry(0, old_x, old_dx, old_p)

    assert builder.exp_idx == 0
    assert builder.fwd == FWDS[0]
    assert np.allclose(builder.strikes, STRIKES[0])
    assert np.allclose(builder.cf_prices, make_prices()[0])


def test_lv_bysections_builder_objective_feasible_params_returns_finite_nonneg():
    """ Feasible params must produce a finite, non-negative RMSE """
    builder = make_builder()
    old_x, old_dx, old_p = builder.initialize()
    builder.set_expiry(0, old_x, old_dx, old_p)

    result = builder.objective(VALID_SVI)

    assert np.isfinite(result)
    assert result >= 0.0


def test_lv_bysections_builder_calculate_vols_has_correct_length_and_positive_values():
    """ calculate_vols() must return one positive vol per strike """
    builder = make_builder()
    old_x, old_dx, old_p = builder.initialize()
    builder.set_expiry(0, old_x, old_dx, old_p)
    builder.objective(VALID_SVI) # Populates pde_prices

    vols = builder.calculate_vols()

    assert len(vols) == len(STRIKES[0])
    assert all(v > 0.0 for v in vols)


##################### Dupire formula ##############################################################
def test_calib_dupire():
    """ Calibrate LV with Dupire """
    # Set IV surface model
    surface = TsSvi1()
    surface.update_params(surface.initial_point())

    # Calibrate
    result = calib_lv_dupire(surface, None, None, points_per_year=2, n_strikes=3)
    lv = result['lv']
    m = result['moneyness']
    t = result['t_grid']
    assert (np.abs(t[1] - 0.002739726027397) < 1e-10)
    m_test = np.asarray(m[1])
    m_ref = np.asarray([0.88482086, 1.00596945, 1.12711805])
    assert np.allclose(m_test, m_ref, 1e-10)
    lv_test = np.asarray(lv[1])
    lv_ref = np.asarray([0.72218576, 0.19823641, 0.18378965])
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
    ref = np.asarray([0.20000181, 0.20001192, 0.1999994])
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


##################### LV by matrix interpolation ##################################################
def test_lv_bymatrix_pchip_def():
    lv = MatrixLocalVol(T_GRID, LOGM_GRID, make_vol_matrix(), interpolation='pchip')
    assert lv.method == 'pchip'
    lv2 = MatrixLocalVol(T_GRID, LOGM_GRID, make_vol_matrix(), interpolation='cubic')
    assert lv2.method == 'cubic'


def test_lv_bymatrix_value_on_grid_nodes():
    """ Values at grid nodes must exactly reproduce the input matrix. """
    lv = make_mlv()
    vol_matrix = make_vol_matrix()
    for i, t in enumerate(T_GRID):
        for j, lm in enumerate(LOGM_GRID):
            assert np.isclose(lv.value(t, lm), vol_matrix[i, j])


def test_lv_bymatrix_value_interior_bilinear_exact():
    """ Linear interpolation on a bilinear surface is exact at any interior point. """
    lv = make_mlv()
    t, lm = 0.75, 0.1
    # expected = 0.20 + 0.02 * t - 0.05 * lm
    expected = lv_func_def(t, lm)
    assert np.isclose(lv.value(t, lm), expected, atol=1e-12)


def test_lv_bymatrix_extrap_below_t():
    """ t below grid must return the same value as t_grid[0]. """
    lv = make_mlv()
    assert np.isclose(lv.value(0.0, 0.0), lv.value(T_GRID[0], 0.0))


def test_lv_bymatrix_extrap_above_t():
    """ t above grid must return the same value as t_grid[-1]. """
    lv = make_mlv()
    assert np.isclose(lv.value(100.0, 0.0), lv.value(T_GRID[-1], 0.0))


def test_lv_bymatrix_extrap_below_logm():
    """ logm below grid must return the same value as logm_grid[0]. """
    lv = make_mlv()
    assert np.isclose(lv.value(1.0, -10.0), lv.value(1.0, LOGM_GRID[0]))


def test_lv_bymatrix_extrap_above_logm():
    """ logm above grid must return the same value as logm_grid[-1]. """
    lv = make_mlv()
    assert np.isclose(lv.value(1.0, 10.0), lv.value(1.0, LOGM_GRID[-1]))


def test_lv_bymatrix_flat_surface_everywhere():
    """ A flat vol surface must return the same value at all (t, logm), including outside the grid. """
    lv = MatrixLocalVol(T_GRID, LOGM_GRID, np.full((4, 5), 0.25))
    for t in [0.0, 0.5, 1.5, 5.0]:
        for lm in [-1.0, 0.0, 1.0]:
            assert np.isclose(lv.value(t, lm), 0.25)


def test_lv_bymatrix_section_consistent_with_value():
    """ section(t)(logm) must equal value(t, logm) for any logm. """
    lv = make_mlv()
    t = 0.75
    logm = np.array([-0.15, 0.0, 0.15])
    assert np.allclose(lv.section(t)(logm), lv.value(t, logm))


##################### LV by sections ##############################################################
def test_lv_sections_sorted_by_time():
    """ Sections passed in reverse order should be stored sorted by time """
    sections = [SviSection(t) for t in [2.0, 0.5, 1.0]]
    lv = InterpolatedParamLocalVol(sections)
    times = [s.time for s in lv.sections]
    assert times == sorted(times)


def test_lv_t_grid_matches_sections():
    """ LV's time grid matches sections' """
    sections = [SviSection(t) for t in [3.0, 1.0, 0.5]]
    lv = InterpolatedParamLocalVol(sections)
    assert lv.t_grid == [0.5, 1.0, 3.0]


def test_lv_by_section_values():
    """Between pillars, value() must delegate to the upper (right) section."""
    t_grid = [0.5, 1.0, 2.0]
    params_mid  = np.array([0.04, 0.1, 0.0, 0.0, 0.2])
    params_high = np.array([0.09, 0.2, 0.0, 0.0, 0.3])
    sections = [SviSection(t) for t in t_grid]
    lv = InterpolatedParamLocalVol(sections)
    lv.update_params(0, params_mid)
    lv.update_params(1, params_high)
    lv.update_params(2, params_mid)

    logm = [-0.5, 0.0, 0.5]
    test = lv.value(0.75, logm)
    ref = np.asarray([0.52487337, 0.4472136,  0.52487337,])
    assert np.allclose(test, ref, 1e-10)


def test_lv_sections_return_time():
    lv = make_lv_by_sections(t_grid=[0.5, 1.0])
    assert lv.section(0).time == 0.5
    assert lv.section(1).time == 1.0


def test_lv_update_params_do_not_mutate():
    """ Mutating the source array after update_params must not change stored params """
    lv = make_lv_by_sections(t_grid=[1.0])
    p = np.array([0.04, 0.1, 0.0, 0.0, 0.2])
    lv.update_params(0, p)
    p[0] = 999.0
    assert lv.params(0)[0] != 999.0


def test_lv_check_params():
    lv = make_lv_by_sections()
    is_ok, penalty = lv.check_params(0)
    assert is_ok
    assert penalty == 0.0


def test_lv_bysections_dump_data_keys():
    data = make_lv_by_sections().dump_data()
    assert set(data.keys()) == {'name', 'valdate', 'snapdate', 'sections'}


if __name__ == "__main__":
    test_calib_dupire()
    # test_dupire_impliedvol()
    # test_dupire_pdf()
