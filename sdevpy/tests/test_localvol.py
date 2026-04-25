import numpy as np
import numpy.typing as npt
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol, MatrixLocalVol
from sdevpy.volatility.impliedvol.models.svi import SviSection


VALID_PARAMS = np.array([0.04, 0.1, 0.0, 0.0, 0.2])  # a, b, rho, m, sigma
T_GRID    = np.array([0.25, 0.5, 1.0, 2.0])
LOGM_GRID = np.array([-0.25, -0.2, 0.0, 0.2, 0.25])


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


def make_lv(**kwargs):
    return MatrixLocalVol(T_GRID, LOGM_GRID, VOL_MATRIX, **kwargs)


VOL_MATRIX = make_vol_matrix()


##################### LV by matrix interpolation ##################################################


def test_lv_bymatrix_pchip_def():
    lv = MatrixLocalVol(T_GRID, LOGM_GRID, VOL_MATRIX, interpolation='pchip')
    assert lv.method == 'pchip'
    lv2 = MatrixLocalVol(T_GRID, LOGM_GRID, VOL_MATRIX, interpolation='cubic')
    assert lv2.method == 'cubic'


def test_lv_bymatrix_value_on_grid_nodes():
    """ Values at grid nodes must exactly reproduce the input matrix. """
    lv = make_lv()
    for i, t in enumerate(T_GRID):
        for j, lm in enumerate(LOGM_GRID):
            assert np.isclose(lv.value(t, lm), VOL_MATRIX[i, j])


def test_lv_bymatrix_value_interior_bilinear_exact():
    """ Linear interpolation on a bilinear surface is exact at any interior point. """
    lv = make_lv()
    t, lm = 0.75, 0.1
    # expected = 0.20 + 0.02 * t - 0.05 * lm
    expected = lv_func_def(t, lm)
    assert np.isclose(lv.value(t, lm), expected, atol=1e-12)


def test_lv_bymatrix_extrap_below_t():
    """ t below grid must return the same value as t_grid[0]. """
    lv = make_lv()
    assert np.isclose(lv.value(0.0, 0.0), lv.value(T_GRID[0], 0.0))


def test_lv_bymatrix_extrap_above_t():
    """ t above grid must return the same value as t_grid[-1]. """
    lv = make_lv()
    assert np.isclose(lv.value(100.0, 0.0), lv.value(T_GRID[-1], 0.0))


def test_lv_bymatrix_extrap_below_logm():
    """ logm below grid must return the same value as logm_grid[0]. """
    lv = make_lv()
    assert np.isclose(lv.value(1.0, -10.0), lv.value(1.0, LOGM_GRID[0]))


def test_lv_bymatrix_extrap_above_logm():
    """ logm above grid must return the same value as logm_grid[-1]. """
    lv = make_lv()
    assert np.isclose(lv.value(1.0, 10.0), lv.value(1.0, LOGM_GRID[-1]))


def test_lv_bymatrix_flat_surface_everywhere():
    """ A flat vol surface must return the same value at all (t, logm), including outside the grid. """
    lv = MatrixLocalVol(T_GRID, LOGM_GRID, np.full((4, 5), 0.25))
    for t in [0.0, 0.5, 1.5, 5.0]:
        for lm in [-1.0, 0.0, 1.0]:
            assert np.isclose(lv.value(t, lm), 0.25)


def test_lv_bymatrix_section_consistent_with_value():
    """ section(t)(logm) must equal value(t, logm) for any logm. """
    lv = make_lv()
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
    # print(test)
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
    test_matrix_lv_custom_method()
