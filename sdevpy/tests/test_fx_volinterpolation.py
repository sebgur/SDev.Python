""" Tests for the two-dimensional FX vol surface interpolation (expiry x delta) """
import pytest
import datetime as dt
import numpy as np
from sdevpy.utilities import dates as dts
from sdevpy.maths.interpolation import create_interpolation
from sdevpy.market.fx.fxconventions import fx_market_yearfraction
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider
from sdevpy.volatility.fx.fx_volinterpolation import FxVolInterpolation, interpolation_from_fxvol_data


VALDATE = dt.datetime(2025, 12, 15)
PAIR = "USDJPY"
DELTAS = [0.10, 0.25, 0.50, 0.75, 0.90]
EXPIRIES = [dt.datetime(2026, 3, 15), dt.datetime(2026, 6, 15), dt.datetime(2026, 12, 15)]
TIMES = np.asarray([fx_market_yearfraction(VALDATE, e) for e in EXPIRIES])

# Three smiles of the same shape (skewed, ATM minimum), each shifted up with expiry
SMILE_SHAPE = [0.120, 0.105, 0.095, 0.100, 0.115]
SHIFTS = [0.000, 0.004, 0.010]
PILLAR_VOLS = [[v + s for v in SMILE_SHAPE] for s in SHIFTS]


def _smile(vols, interp='linear', extrap='flat'):
    """ One delta-direction interpolation on the standard put-delta grid """
    return create_interpolation(interp=interp, l_extrap=extrap, r_extrap=extrap,
                                x_grid=list(DELTAS), y_grid=list(vols))


def _surface(**kwargs):
    """ Reference surface: three pillars, upward-shifting smiles """
    return FxVolInterpolation(VALDATE, list(EXPIRIES), [_smile(v) for v in PILLAR_VOLS], **kwargs)


def _flat_surface(vol=0.10, **kwargs):
    """ Same vol at every delta and every pillar. Total variance is then exactly linear in t
        through the origin, so every mode must return `vol` everywhere, interpolated or not. """
    smiles = [_smile([vol] * len(DELTAS)) for _ in EXPIRIES]
    return FxVolInterpolation(VALDATE, list(EXPIRIES), smiles, **kwargs)


def _pillar_vol(idx, delta):
    """ Vol read straight off pillar idx's own smile, bypassing the surface """
    return float(_smile(PILLAR_VOLS[idx]).value(delta))


class TestConstruction:
    def test_pillars_are_sorted_by_expiry(self):
        order = [2, 0, 1]
        surface = FxVolInterpolation(VALDATE, [EXPIRIES[i] for i in order],
                                     [_smile(PILLAR_VOLS[i]) for i in order])
        assert surface.expiries == EXPIRIES
        assert np.all(np.diff(surface.times) > 0.0)

    def test_shuffled_pillars_keep_their_own_smiles(self):
        """ Sorting must permute expiries and interpolations together, not just the expiries """
        order = [2, 0, 1]
        surface = FxVolInterpolation(VALDATE, [EXPIRIES[i] for i in order],
                                     [_smile(PILLAR_VOLS[i]) for i in order])
        for i, expiry in enumerate(EXPIRIES):
            assert float(surface.vol_at_delta(expiry, 0.25)) == pytest.approx(PILLAR_VOLS[i][1])

    def test_times_follow_the_fx_market_yearfraction(self):
        assert _surface().times == pytest.approx(TIMES)

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, EXPIRIES, [_smile(PILLAR_VOLS[0])])

    def test_no_pillar_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [], [])

    def test_unknown_time_interp_raises(self):
        with pytest.raises(ValueError):
            _surface(time_interp='cubic')

    def test_unknown_time_extrap_raises(self):
        with pytest.raises(ValueError):
            _surface(time_extrap='quadratic')

    def test_expiry_on_valuation_date_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [VALDATE], [_smile(PILLAR_VOLS[0])])

    def test_expiry_before_valuation_date_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [dt.datetime(2025, 6, 15)], [_smile(PILLAR_VOLS[0])])

    def test_duplicate_expiries_raise(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [EXPIRIES[0], EXPIRIES[0]],
                               [_smile(PILLAR_VOLS[0]), _smile(PILLAR_VOLS[1])])


class TestDeltaDirection:
    def test_quoted_deltas_are_reproduced_at_every_pillar(self):
        surface = _surface()
        for i, expiry in enumerate(EXPIRIES):
            assert surface.vol_at_delta(expiry, DELTAS) == pytest.approx(PILLAR_VOLS[i])

    def test_smile_keeps_its_shape_between_quoted_deltas(self):
        """ Put wing above ATM, call wing above ATM: the interpolated smile stays a smile """
        surface = _surface()
        vols = surface.vol_at_delta(EXPIRIES[0], [0.15, 0.35, 0.50, 0.65, 0.85])
        assert vols[0] > vols[1] > vols[2]
        assert vols[2] < vols[3] < vols[4]

    def test_delta_extrapolation_is_flat_outside_the_quoted_range(self):
        surface = _surface()
        assert float(surface.vol_at_delta(EXPIRIES[0], 0.01)) == pytest.approx(PILLAR_VOLS[0][0])
        assert float(surface.vol_at_delta(EXPIRIES[0], 0.99)) == pytest.approx(PILLAR_VOLS[0][-1])


class TestTimeDirection:
    @pytest.mark.parametrize('time_interp', ['var', 'vol2', 'vol'])
    def test_pillar_expiries_return_pillar_vols(self, time_interp):
        """ Whatever the time mode, landing exactly on a pillar must return that pillar's smile """
        surface = _surface(time_interp=time_interp)
        for i, expiry in enumerate(EXPIRIES):
            assert surface.vol_at_delta(expiry, DELTAS) == pytest.approx(PILLAR_VOLS[i])

    def test_var_mode_interpolates_linearly_in_total_variance(self):
        surface = _surface(time_interp='var')
        delta, t = 0.25, 0.5 * (TIMES[0] + TIMES[1])
        s0, s1 = _pillar_vol(0, delta), _pillar_vol(1, delta)
        w0, w1 = s0 ** 2 * TIMES[0], s1 ** 2 * TIMES[1]
        theta = (t - TIMES[0]) / (TIMES[1] - TIMES[0])
        expected = np.sqrt((w0 + theta * (w1 - w0)) / t)
        assert float(surface.vol_at_delta(t, delta)) == pytest.approx(expected)

    def test_vol2_mode_interpolates_linearly_in_variance(self):
        surface = _surface(time_interp='vol2')
        delta, t = 0.25, 0.5 * (TIMES[0] + TIMES[1])
        s0, s1 = _pillar_vol(0, delta), _pillar_vol(1, delta)
        theta = (t - TIMES[0]) / (TIMES[1] - TIMES[0])
        expected = np.sqrt(s0 ** 2 + theta * (s1 ** 2 - s0 ** 2))
        assert float(surface.vol_at_delta(t, delta)) == pytest.approx(expected)

    def test_vol_mode_interpolates_linearly_in_vol(self):
        surface = _surface(time_interp='vol')
        delta, t = 0.25, 0.5 * (TIMES[0] + TIMES[1])
        s0, s1 = _pillar_vol(0, delta), _pillar_vol(1, delta)
        theta = (t - TIMES[0]) / (TIMES[1] - TIMES[0])
        assert float(surface.vol_at_delta(t, delta)) == pytest.approx(s0 + theta * (s1 - s0))

    @pytest.mark.parametrize('time_interp', ['var', 'vol2', 'vol'])
    def test_flat_term_structure_is_preserved(self, time_interp):
        """ Constant vol across pillars must stay constant everywhere in between. For 'var' this
            is the non-trivial one: total variance has to be exactly linear through the origin. """
        surface = _flat_surface(0.10, time_interp=time_interp)
        t = np.linspace(TIMES[0], TIMES[-1], 25)
        assert surface.vol_at_delta(t, 0.25) == pytest.approx(0.10)

    def test_interpolated_vol_lies_between_the_two_pillar_vols(self):
        surface = _surface()
        delta = 0.25
        t = np.linspace(TIMES[0], TIMES[1], 20)[1:-1]
        vols = surface.vol_at_delta(t, delta)
        assert np.all(vols > _pillar_vol(0, delta))
        assert np.all(vols < _pillar_vol(1, delta))

    def test_term_structure_is_increasing_for_increasing_smiles(self):
        surface = _surface()
        t = np.linspace(TIMES[0], TIMES[-1], 50)
        assert np.all(np.diff(surface.vol_at_delta(t, 0.25)) > 0.0)


class TestTimeExtrapolation:
    @pytest.mark.parametrize('time_extrap', ['flat', 'linear'])
    def test_before_first_pillar_is_always_flat_in_vol(self, time_extrap):
        """ Short end is flat whatever time_extrap says: the only line in total variance that
            reaches t=0 without going negative is the one through the origin, i.e. flat vol """
        surface = _surface(time_extrap=time_extrap)
        short = np.asarray([0.001, 0.01, 0.05, 0.5 * TIMES[0]])
        assert surface.vol_at_delta(short, 0.25) == pytest.approx(_pillar_vol(0, 0.25))

    @pytest.mark.parametrize('time_extrap', ['flat', 'linear'])
    def test_short_end_never_collapses_to_zero_vol(self, time_extrap):
        """ Regression: backward extrapolation of total variance used to go negative, get floored
            at zero and silently return a zero vol """
        surface = _surface(time_extrap=time_extrap)
        vols = surface.vol_at_delta(np.asarray([1e-4, 1e-3, 1e-2]), DELTAS[:3])
        assert np.all(vols > 0.01)

    def test_after_last_pillar_is_flat_when_requested(self):
        surface = _surface(time_extrap='flat')
        far = np.asarray([TIMES[-1] + 0.1, 5.0, 30.0])
        assert surface.vol_at_delta(far, 0.25) == pytest.approx(_pillar_vol(2, 0.25))

    def test_after_last_pillar_linear_follows_the_total_variance_line(self):
        surface = _surface(time_extrap='linear', time_interp='var')
        delta, t = 0.25, TIMES[-1] + 1.0
        s0, s1 = _pillar_vol(1, delta), _pillar_vol(2, delta)
        w0, w1 = s0 ** 2 * TIMES[-2], s1 ** 2 * TIMES[-1]
        theta = (t - TIMES[-2]) / (TIMES[-1] - TIMES[-2])
        expected = np.sqrt((w0 + theta * (w1 - w0)) / t)
        assert float(surface.vol_at_delta(t, delta)) == pytest.approx(expected)

    def test_linear_extrapolation_differs_from_flat_beyond_the_last_pillar(self):
        t = TIMES[-1] + 2.0
        flat = float(_surface(time_extrap='flat').vol_at_delta(t, 0.25))
        linear = float(_surface(time_extrap='linear').vol_at_delta(t, 0.25))
        assert flat == pytest.approx(_pillar_vol(2, 0.25))
        assert linear > flat # rising term structure keeps rising

    def test_linear_extrapolation_preserves_a_flat_term_structure(self):
        surface = _flat_surface(0.10, time_extrap='linear')
        assert surface.vol_at_delta(np.asarray([0.001, 5.0, 50.0]), 0.25) == pytest.approx(0.10)


class TestVectorization:
    def test_scalar_inputs_give_a_scalar_result(self):
        v = _surface().vol_at_delta(EXPIRIES[0], 0.25)
        assert np.shape(v) == ()
        assert float(v) == pytest.approx(PILLAR_VOLS[0][1])

    def test_delta_vector_at_a_single_expiry(self):
        v = _surface().vol_at_delta(EXPIRIES[0], DELTAS)
        assert v.shape == (len(DELTAS),)

    def test_expiry_vector_at_a_single_delta(self):
        v = _surface().vol_at_delta(EXPIRIES, 0.25)
        assert v.shape == (len(EXPIRIES),)

    def test_broadcasting_gives_the_outer_grid(self):
        surface = _surface()
        t = np.linspace(0.05, 3.0, 7)
        d = np.linspace(0.05, 0.95, 4)
        v = surface.vol_at_delta(t[:, None], d[None, :])
        assert v.shape == (7, 4)

    @pytest.mark.parametrize('time_extrap', ['flat', 'linear'])
    def test_vectorized_call_matches_the_scalar_loop(self, time_extrap):
        """ Points deliberately spread over every bracket and both extrapolation regions, so the
            per-pillar grouping in _smile_values has to dispatch each point to the right smile """
        surface = _surface(time_extrap=time_extrap)
        t = np.asarray([0.01, 0.1, TIMES[0], 0.4, TIMES[1], 0.8, TIMES[2], 1.5, 9.0])
        d = np.asarray([0.05, 0.10, 0.22, 0.40, 0.50, 0.60, 0.78, 0.90, 0.97])
        vectorized = surface.vol_at_delta(t, d)
        scalar = [float(surface.vol_at_delta(ti, di)) for ti, di in zip(t, d)]
        assert vectorized == pytest.approx(scalar)

    def test_year_fractions_and_dates_agree(self):
        surface = _surface()
        assert surface.vol_at_delta(TIMES, 0.25) == pytest.approx(surface.vol_at_delta(EXPIRIES, 0.25))

    def test_vol_grid_matches_pointwise_evaluation(self):
        surface = _surface()
        t = np.asarray([0.3, 0.7, 1.4])
        grid = surface.vol_grid(t, DELTAS)
        assert grid.shape == (len(t), len(DELTAS))
        for i, ti in enumerate(t):
            assert grid[i] == pytest.approx(surface.vol_at_delta(ti, DELTAS))

    def test_vol_grid_at_pillars_returns_the_pillar_smiles(self):
        grid = _surface().vol_grid(EXPIRIES, DELTAS)
        assert grid == pytest.approx(np.asarray(PILLAR_VOLS))


class TestVariance:
    def test_var_at_delta_is_vol_squared_times_time(self):
        surface = _surface()
        t = np.asarray([0.3, 0.7, 1.4])
        expected = surface.vol_at_delta(t, 0.25) ** 2 * t
        assert surface.var_at_delta(t, 0.25) == pytest.approx(expected)

    def test_var_at_delta_accepts_dates(self):
        surface = _surface()
        expected = np.asarray(PILLAR_VOLS)[:, 1] ** 2 * TIMES
        assert surface.var_at_delta(EXPIRIES, 0.25) == pytest.approx(expected)

    def test_total_variance_increases_with_expiry(self):
        surface = _surface()
        t = np.linspace(TIMES[0], TIMES[-1], 40)
        assert np.all(np.diff(surface.var_at_delta(t, 0.25)) > 0.0)


class TestCalendarCheck:
    def test_increasing_surface_passes(self):
        assert _surface().calendar_check() is True

    def test_flat_surface_passes(self):
        assert _flat_surface().calendar_check() is True

    def test_decreasing_total_variance_is_flagged(self, caplog):
        """ A far pillar whose total variance falls below the near one is calendar arbitrage """
        vols = [[0.30] * len(DELTAS), [0.05] * len(DELTAS), [0.05] * len(DELTAS)]
        surface = FxVolInterpolation(VALDATE, list(EXPIRIES), [_smile(v) for v in vols])
        with caplog.at_level('WARNING'):
            assert surface.calendar_check() is False
        assert 'Calendar arbitrage' in caplog.text


class TestSinglePillar:
    def test_single_pillar_surface_is_flat_in_time(self):
        surface = FxVolInterpolation(VALDATE, [EXPIRIES[1]], [_smile(PILLAR_VOLS[1])])
        t = np.asarray([0.01, 0.5, TIMES[1], 5.0])
        assert surface.vol_at_delta(t, 0.25) == pytest.approx(PILLAR_VOLS[1][1])

    def test_single_pillar_surface_keeps_its_smile(self):
        surface = FxVolInterpolation(VALDATE, [EXPIRIES[1]], [_smile(PILLAR_VOLS[1])])
        assert surface.vol_at_delta(3.0, DELTAS) == pytest.approx(PILLAR_VOLS[1])


class TestFromCalibratedData:
    @staticmethod
    def _data():
        return CalibrationDataFileProvider().get_fxvol_data(PAIR, VALDATE)

    def test_builds_one_pillar_per_tenor_report(self):
        data = self._data()
        surface = interpolation_from_fxvol_data(data)
        assert len(surface.expiries) == len(data['tenor_reports'])
        assert surface.valdate == VALDATE

    def test_pillar_expiries_match_the_file(self):
        data = self._data()
        surface = interpolation_from_fxvol_data(data)
        expected = sorted(dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
                          for r in data['tenor_reports'])
        assert surface.expiries == expected

    def test_market_vols_are_reproduced_at_every_quoted_point(self):
        data = self._data()
        surface = interpolation_from_fxvol_data(data, smile_interp='linear', smile_extrap='flat')
        for report in data['tenor_reports']:
            expiry = dt.datetime.strptime(report['expiry'], dts.DATE_FILE_FORMAT)
            assert surface.vol_at_delta(expiry, report['deltas']) == pytest.approx(report['vols'])

    def test_calendar_check_flags_the_1m_dip_in_the_test_data(self, caplog):
        """ The stored USDJPY set is mostly one repeated smile, with 1W and 1M carrying different
            quotes. The 1M smile sits below the block, so total variance falls from 3W to 1M on
            the call wing: a real (small) calendar arbitrage in the data, correctly detected. """
        surface = interpolation_from_fxvol_data(self._data())
        with caplog.at_level('WARNING'):
            assert surface.calendar_check() is False
        assert 'between pillars 2 and 3' in caplog.text

    def test_calendar_check_passes_away_from_the_1m_pillar(self):
        """ Same data, dropping the 1M tenor: the rest of the term structure is arbitrage-free """
        data = self._data()
        data['tenor_reports'] = [r for r in data['tenor_reports'] if r['tenor'] != '1M']
        assert interpolation_from_fxvol_data(data).calendar_check() is True

    def test_off_pillar_expiry_stays_within_the_neighbouring_vols(self):
        data = self._data()
        surface = interpolation_from_fxvol_data(data)
        reports = sorted(data['tenor_reports'], key=lambda r: r['expiry'])
        e0 = dt.datetime.strptime(reports[3]['expiry'], dts.DATE_FILE_FORMAT)
        e1 = dt.datetime.strptime(reports[4]['expiry'], dts.DATE_FILE_FORMAT)
        mid = e0 + (e1 - e0) / 2
        v0 = float(surface.vol_at_delta(e0, 0.25))
        v1 = float(surface.vol_at_delta(e1, 0.25))
        vmid = float(surface.vol_at_delta(mid, 0.25))
        assert min(v0, v1) <= vmid <= max(v0, v1)

    def test_whole_market_surface_evaluates_in_one_vectorized_call(self):
        surface = interpolation_from_fxvol_data(self._data())
        t = np.linspace(0.01, 5.0, 60)
        d = np.linspace(0.02, 0.98, 40)
        vols = surface.vol_at_delta(t[:, None], d[None, :])
        assert vols.shape == (60, 40)
        assert np.all(np.isfinite(vols))
        assert np.all(vols > 0.0)
