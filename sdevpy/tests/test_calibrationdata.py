""" Tests for the calibration data layer: source, dataset (cache + save) and repository (LRU) """
import copy
import pytest
import datetime as dt
import numpy as np
from sdevpy.pricingcontext import default_calibration_repository
from sdevpy.calibration.source import CalibrationDataSource
from sdevpy.calibration.filesource import CalibrationDataFileSource
from sdevpy.calibration.dataset import CalibrationDataSet
from sdevpy.calibration.repository import CalibrationDataRepository


VALDATE = dt.datetime(2025, 12, 15)
REF_CALIB = default_calibration_repository()[VALDATE] # Stored test data, read only
BIEXP_DATA = REF_CALIB.get_localvol_data('ABC', 'BiExp')
LOGMIX_DATA = REF_CALIB.get_impliedvol_data('ABC', 'LogMix3')


class FakeCurve:
    def __init__(self, name):
        self.name = name


class InMemorySource(CalibrationDataSource):
    """ Test double: stores data in dicts and counts reads, so tests can check caching """
    def __init__(self, lv=None, iv=None, fx=None):
        self.lv = dict(lv or {})
        self.iv = dict(iv or {})
        self.fx = dict(fx or {})
        self.reads = []

    def get_yieldcurve(self, name, date):
        self.reads.append(('yc', name, date))
        return FakeCurve(name)

    def get_impliedvol_data(self, name, date, model_name):
        self.reads.append(('iv', name, date, model_name))
        return self.iv.get((name, date, model_name))

    def get_localvol_data(self, name, date, model_name):
        self.reads.append(('lv', name, date, model_name))
        return self.lv.get((name, date, model_name))

    def get_fxvol_data(self, pair, date):
        self.reads.append(('fx', pair, date))
        return self.fx.get((pair, date))

    def save_impliedvol_data(self, name, date, model_name, data):
        self.iv[(name, date, model_name)] = data

    def save_localvol_data(self, name, date, model_name, data):
        self.lv[(name, date, model_name)] = data

    def save_fxvol_data(self, pair, date, data):
        self.fx[(pair, date)] = data


def _biexp_calib():
    source = InMemorySource(lv={('ABC', VALDATE, 'BiExp'): copy.deepcopy(BIEXP_DATA)})
    return CalibrationDataSet(VALDATE, source), source


##################### Dataset: caching ############################################################

class TestDatasetCache:
    def test_yieldcurve_is_loaded_once(self):
        source = InMemorySource()
        calib = CalibrationDataSet(VALDATE, source)
        assert calib.get_yieldcurve('USD.SOFR.1D') is calib.get_yieldcurve('USD.SOFR.1D')
        assert source.reads == [('yc', 'USD.SOFR.1D', VALDATE)]

    def test_rfr_and_xccy_share_the_same_cached_curve(self):
        source = InMemorySource()
        calib = CalibrationDataSet(VALDATE, source)
        assert calib.get_xccycurve('USD') is calib.get_rfrcurve('USD')
        assert len(source.reads) == 1

    def test_missing_data_is_cached_as_none(self):
        source = InMemorySource()
        calib = CalibrationDataSet(VALDATE, source)
        assert calib.get_localvol_data('ABC', 'BiExp') is None
        assert calib.get_localvol_data('ABC', 'BiExp') is None
        assert len(source.reads) == 1

    def test_cache_is_keyed_on_model_name(self):
        calib, source = _biexp_calib()
        assert calib.get_localvol_data('ABC', 'BiExp') is not None
        assert calib.get_localvol_data('ABC', 'VSVI') is None
        assert len(source.reads) == 2

    def test_clear_forces_a_reload(self):
        source = InMemorySource()
        calib = CalibrationDataSet(VALDATE, source)
        calib.get_yieldcurve('USD.SOFR.1D')
        calib.clear()
        calib.get_yieldcurve('USD.SOFR.1D')
        assert len(source.reads) == 2


##################### Dataset: vol models are built fresh #########################################

class TestDatasetModels:
    def test_localvol_is_a_new_object_on_every_call(self):
        calib, source = _biexp_calib()
        assert calib.get_localvol('ABC', 'BiExp') is not calib.get_localvol('ABC', 'BiExp')
        assert len(source.reads) == 1 # Built twice from one cached load

    def test_mutating_a_localvol_does_not_touch_the_cache(self):
        calib, _ = _biexp_calib()
        lv = calib.get_localvol('ABC', 'BiExp')
        lv.update_params(0, np.asarray(lv.params(0)) * 2.0)
        assert calib.get_localvol_data('ABC', 'BiExp') == BIEXP_DATA
        fresh = calib.get_localvol('ABC', 'BiExp')
        assert np.allclose(fresh.params(0), np.asarray(lv.params(0)) / 2.0)

    def test_impliedvol_is_a_new_object_on_every_call(self):
        source = InMemorySource(iv={('ABC', VALDATE, 'LogMix3'): copy.deepcopy(LOGMIX_DATA)})
        calib = CalibrationDataSet(VALDATE, source)
        assert calib.get_impliedvol('ABC', 'LogMix3') is not calib.get_impliedvol('ABC', 'LogMix3')

    def test_localvol_on_new_time_grid(self):
        calib, _ = _biexp_calib()
        t_grid = [0.0, 0.5, 1.0, 1.5]
        lv = calib.get_localvol('ABC', 'BiExp', t_grid=t_grid)
        assert lv.t_grid == t_grid


##################### Dataset: get_localvol_or_new / get_local_vols ###############################

class TestLocalVolOrNew:
    def test_model_name_inferred_from_name_model_map(self):
        calib, _ = _biexp_calib() # ABC -> BiExp in name_model_map
        lv = calib.get_localvol_or_new('ABC', None)
        assert lv.dump_data()['sections'] == BIEXP_DATA['sections']

    def test_unknown_name_without_model_raises(self):
        calib, _ = _biexp_calib()
        with pytest.raises(ValueError, match="No model name"):
            calib.get_localvol_or_new('NOPE', None)

    def test_missing_model_without_t_grid_raises(self):
        calib = CalibrationDataSet(VALDATE, InMemorySource())
        with pytest.raises(ValueError, match="no t_grid"):
            calib.get_localvol_or_new('ABC', 'BiExp')

    def test_missing_model_with_t_grid_builds_new(self):
        calib = CalibrationDataSet(VALDATE, InMemorySource())
        lv = calib.get_localvol_or_new('ABC', 'BiExp', t_grid=[0.0, 1.0])
        assert lv.t_grid == [0.0, 1.0]

    def test_force_new_ignores_stored_model(self):
        calib, source = _biexp_calib()
        lv = calib.get_localvol_or_new('ABC', 'BiExp', t_grid=[0.0, 1.0], force_new=True)
        assert lv.t_grid == [0.0, 1.0]
        assert source.reads == [] # Never looked at the store

    def test_local_vols_from_lv_map(self):
        calib = CalibrationDataSet(VALDATE, InMemorySource())
        lv = object()
        assert calib.get_local_vols(['ABC'], lv_map={'ABC': lv}) == [lv]

    def test_local_vols_missing_from_lv_map_raises(self):
        calib = CalibrationDataSet(VALDATE, InMemorySource())
        with pytest.raises(ValueError, match="Could not find LV"):
            calib.get_local_vols(['ABC'], lv_map={})


##################### Dataset: saving #############################################################

class TestDatasetSave:
    def test_save_replaces_a_cached_none(self):
        source = InMemorySource()
        calib = CalibrationDataSet(VALDATE, source)
        assert calib.get_localvol_data('ABC', 'BiExp') is None # Cached
        calib.save_localvol_data('ABC', 'BiExp', copy.deepcopy(BIEXP_DATA))
        assert calib.get_localvol_data('ABC', 'BiExp') == BIEXP_DATA

    def test_save_replaces_cached_data(self):
        source = InMemorySource(fx={('USDJPY', VALDATE): {'v': 1}})
        calib = CalibrationDataSet(VALDATE, source)
        assert calib.get_fxvol_data('USDJPY') == {'v': 1}
        calib.save_fxvol_data('USDJPY', {'v': 2})
        assert calib.get_fxvol_data('USDJPY') == {'v': 2}

    def test_save_localvol_stores_its_dump(self):
        calib, source = _biexp_calib()
        lv = calib.get_localvol('ABC', 'BiExp')
        lv.name, lv.valdate, lv.snapdate = 'ABC', VALDATE, VALDATE
        calib.save_localvol('ABC', 'VSVI', lv) # Any model key: checks the write path
        assert source.lv[('ABC', VALDATE, 'VSVI')] == lv.dump_data()

    def test_save_impliedvol_stores_its_dump(self):
        source = InMemorySource()
        calib = CalibrationDataSet(VALDATE, source)
        ivol = REF_CALIB.get_impliedvol('ABC', 'LogMix3')
        calib.save_impliedvol('ABC', 'LogMix3', ivol)
        assert source.iv[('ABC', VALDATE, 'LogMix3')] == ivol.dump_data()
        assert calib.get_impliedvol_data('ABC', 'LogMix3') == ivol.dump_data()


##################### File source #################################################################

class TestFileSource:
    def test_missing_file_returns_none_without_creating_folders(self, tmp_path):
        source = CalibrationDataFileSource(tmp_path)
        assert source.get_localvol_data('ABC', VALDATE, 'BiExp') is None
        assert source.get_impliedvol_data('ABC', VALDATE, 'LogMix3') is None
        assert source.get_fxvol_data('USDJPY', VALDATE) is None
        assert list(tmp_path.iterdir()) == []

    def test_save_writes_the_expected_layout(self, tmp_path):
        source = CalibrationDataFileSource(tmp_path)
        source.save_localvol_data('ABC', VALDATE, 'BiExp', {'a': 1})
        source.save_impliedvol_data('ABC', VALDATE, 'LogMix3', {'b': 2})
        source.save_fxvol_data('USDJPY', VALDATE, {'c': 3})
        files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.json'))
        assert files == ['fxvol/USDJPY/20251215-000000.json',
                         'impliedvol/ABC/20251215-000000.LogMix3.json',
                         'localvol/ABC/20251215-000000.BiExp.json']

    def test_save_overwrites(self, tmp_path):
        source = CalibrationDataFileSource(tmp_path)
        source.save_fxvol_data('USDJPY', VALDATE, {'v': 1})
        source.save_fxvol_data('USDJPY', VALDATE, {'v': 2})
        assert source.get_fxvol_data('USDJPY', VALDATE) == {'v': 2}

    def test_stored_test_data_is_read_from_default_root(self):
        assert BIEXP_DATA is not None and LOGMIX_DATA is not None
        assert REF_CALIB.get_fxvol_data('USDJPY') is not None
        assert REF_CALIB.get_yieldcurve('USD.SOFR.1D') is not None


##################### End-to-end round trips on disk ##############################################

class TestRoundTrip:
    @staticmethod
    def _calib(tmp_path):
        return CalibrationDataRepository(CalibrationDataFileSource(tmp_path))[VALDATE]

    def test_localvol_round_trip(self, tmp_path):
        lv = REF_CALIB.get_localvol('ABC', 'BiExp')
        lv.name, lv.valdate, lv.snapdate = 'ABC', VALDATE, VALDATE
        calib = self._calib(tmp_path)
        assert calib.get_localvol_data('ABC', 'BiExp') is None
        calib.save_localvol('ABC', 'BiExp', lv)
        assert calib.get_localvol('ABC', 'BiExp').dump_data() == lv.dump_data()

    def test_impliedvol_round_trip(self, tmp_path):
        ivol = REF_CALIB.get_impliedvol('ABC', 'LogMix3')
        calib = self._calib(tmp_path)
        calib.save_impliedvol('ABC', 'LogMix3', ivol)
        assert calib.get_impliedvol('ABC', 'LogMix3').dump_data() == ivol.dump_data()

    def test_fxvol_round_trip(self, tmp_path):
        data = copy.deepcopy(REF_CALIB.get_fxvol_data('USDJPY'))
        calib = self._calib(tmp_path)
        calib.save_fxvol_data('USDJPY', data)
        assert calib.get_fxvol_data('USDJPY') == data
