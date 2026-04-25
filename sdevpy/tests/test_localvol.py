import json
import numpy as np
import numpy.typing as npt
from sdevpy.volatility.localvol.localvol import InterpolatedParamLocalVol
from sdevpy.volatility.impliedvol.models.svi import SviSection


VALID_PARAMS = np.array([0.04, 0.1, 0.0, 0.0, 0.2])  # a, b, rho, m, sigma


def make_lv(t_grid: list[float]=None, params: npt.ArrayLike=None):
    if t_grid is None:
        t_grid = [0.25, 0.5, 1.0, 2.0]
    if params is None:
        params = VALID_PARAMS
    sections = [SviSection(t) for t in t_grid]
    lv = InterpolatedParamLocalVol(sections)
    for i in range(len(t_grid)):
        lv.update_params(i, params)
    return lv


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
    lv = make_lv(t_grid=[0.5, 1.0])
    assert lv.section(0).time == 0.5
    assert lv.section(1).time == 1.0


def test_lv_update_params_do_not_mutate():
    """ Mutating the source array after update_params must not change stored params """
    lv = make_lv(t_grid=[1.0])
    p = np.array([0.04, 0.1, 0.0, 0.0, 0.2])
    lv.update_params(0, p)
    p[0] = 999.0
    assert lv.params(0)[0] != 999.0


def test_lv_check_params():
    lv = make_lv()
    is_ok, penalty = lv.check_params(0)
    assert is_ok
    assert penalty == 0.0


def test_lv_bysections_dump_data_keys():
    data = make_lv().dump_data()
    assert set(data.keys()) == {'name', 'valdate', 'snapdate', 'sections'}


if __name__ == "__main__":
    test_lv_by_section_values()
