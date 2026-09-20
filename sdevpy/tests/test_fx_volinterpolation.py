""" Tests for the FX vol surface interpolation (expiry x log-forward-moneyness) """
import pytest
import datetime as dt
import numpy as np
from sdevpy.utilities import dates as dts
from sdevpy.maths.interpolation import create_interpolation
from sdevpy.market.fx import fxconventions
from sdevpy.market.fx.fxforward import fx_spot_date
from sdevpy.market.fx.fxvolsurface import fx_option_dates
from sdevpy.market.fx.fxconventions import fx_market_yearfraction
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.calibration.fileprovider import CalibrationDataFileProvider
from sdevpy.volatility.fx.fx_volinterpolation import FxVolInterpolation, interpolation_from_fxvol_data


PAIR = "USDJPY"
VALDATE = dt.datetime(2025, 12, 15)
FORCCY, DOMCCY = fxconventions.conventional_pair_name(*fxconventions.parse_fx_pair(PAIR))
MD_PROV = MarketDataFileProvider()
SPOT = MD_PROV.get_fx_spot(FORCCY, DOMCCY, VALDATE)
FORCURVE = MD_PROV.get_xccycurve(FORCCY, VALDATE)
DOMCURVE = MD_PROV.get_xccycurve(DOMCCY, VALDATE)

EXPIRIES = [dt.datetime(2026, 3, 16), dt.datetime(2026, 6, 15), dt.datetime(2026, 12, 15)]
TIMES = np.asarray([fx_market_yearfraction(VALDATE, e) for e in EXPIRIES])

# Smile on log-forward-moneyness: skewed, minimum near the forward, shifted up with expiry
MONEYNESS = np.asarray([-0.12, -0.06, 0.00, 0.05, 0.10])
SMILE_SHAPE = np.asarray([0.120, 0.105, 0.095, 0.100, 0.115])
SHIFTS = [0.000, 0.004, 0.010]
PILLAR_VOLS = [list(SMILE_SHAPE + s) for s in SHIFTS]


def _delivery_fwd(expiry):
    """ The delivery-date forward, computed the way the calibrator does """
    settle = fx_spot_date(expiry, FORCCY, DOMCCY)
    return SPOT * FORCURVE.discount(settle) / DOMCURVE.discount(settle)


FWDS = [_delivery_fwd(e) for e in EXPIRIES]


def _smile(vols, interp='linear', extrap='flat'):
    return create_interpolation(interp=interp, l_extrap=extrap, r_extrap=extrap,
                                x_grid=MONEYNESS.copy(), y_grid=list(vols))


def _market(**kwargs):
    base = dict(pair=PAIR, spot=SPOT, forcurve=FORCURVE, domcurve=DOMCURVE)
    base.update(kwargs)
    return base


def _surface(**kwargs):
    """ Reference surface: three pillars, upward-shifting smiles, real curves """
    return FxVolInterpolation(VALDATE, list(EXPIRIES), [_smile(v) for v in PILLAR_VOLS],
                              list(FWDS), **_market(), **kwargs)


def _flat_surface(vol=0.10, **kwargs):
    """ Same vol at every moneyness and every pillar. Total variance is then exactly linear in t
        through the origin, so every mode must return `vol` everywhere. """
    smiles = [_smile([vol] * len(MONEYNESS)) for _ in EXPIRIES]
    return FxVolInterpolation(VALDATE, list(EXPIRIES), smiles, list(FWDS), **_market(), **kwargs)


def _pillar_vol(idx, m):
    return float(_smile(PILLAR_VOLS[idx]).value(m))


class TestConstruction:
    def test_pillars_are_sorted_by_expiry(self):
        o = [2, 0, 1]
        s = FxVolInterpolation(VALDATE, [EXPIRIES[i] for i in o],
                               [_smile(PILLAR_VOLS[i]) for i in o], [FWDS[i] for i in o], **_market())
        assert s.expiries == EXPIRIES
        assert np.all(np.diff(s.times) > 0.0)

    def test_shuffled_pillars_keep_their_own_smiles_and_forwards(self):
        o = [2, 0, 1]
        s = FxVolInterpolation(VALDATE, [EXPIRIES[i] for i in o],
                               [_smile(PILLAR_VOLS[i]) for i in o], [FWDS[i] for i in o], **_market())
        assert s.fwds == pytest.approx(FWDS)
        for i, e in enumerate(EXPIRIES):
            assert float(s.vol_at_moneyness(e, 0.0)) == pytest.approx(PILLAR_VOLS[i][2])

    def test_times_follow_the_fx_market_yearfraction(self):
        assert _surface().times == pytest.approx(TIMES)

    def test_conventions_are_derived_from_the_pair(self):
        s = _surface()
        assert (s.forccy, s.domccy) == (FORCCY, DOMCCY)
        assert s.prem_adj is fxconventions.is_premium_adjusted(FORCCY, DOMCCY)

    def test_size_mismatch_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, EXPIRIES, [_smile(PILLAR_VOLS[0])], FWDS, **_market())

    def test_forward_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, EXPIRIES, [_smile(v) for v in PILLAR_VOLS],
                               FWDS[:2], **_market())

    def test_no_pillar_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [], [], [], **_market())

    def test_non_positive_forward_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, EXPIRIES, [_smile(v) for v in PILLAR_VOLS],
                               [FWDS[0], -1.0, FWDS[2]], **_market())

    @pytest.mark.parametrize('kw', [{'time_interp': 'cubic'}, {'time_extrap': 'quadratic'}])
    def test_unknown_time_mode_raises(self, kw):
        with pytest.raises(ValueError):
            _surface(**kw)

    def test_expiry_on_valuation_date_raises(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [VALDATE], [_smile(PILLAR_VOLS[0])], [FWDS[0]], **_market())

    def test_duplicate_expiries_raise(self):
        with pytest.raises(ValueError):
            FxVolInterpolation(VALDATE, [EXPIRIES[0], EXPIRIES[0]],
                               [_smile(PILLAR_VOLS[0]), _smile(PILLAR_VOLS[1])],
                               [FWDS[0], FWDS[0]], **_market())


class TestForward:
    def test_settlement_is_the_spot_date_after_expiry(self):
        s = _surface()
        assert s.settlement(EXPIRIES[0]) == fx_spot_date(EXPIRIES[0], FORCCY, DOMCCY)

    def test_settlement_matches_the_calibrator_delivery_date(self):
        """ The calibration stored fwd off fx_option_dates' delivery; the surface must agree """
        s = _surface()
        expiry, delivery = fx_option_dates(VALDATE, '1Y', FORCCY, DOMCCY)
        assert s.settlement(expiry) == delivery

    def test_forward_reproduces_the_pillar_forwards(self):
        s = _surface()
        assert s.forward(np.asarray(EXPIRIES, dtype=object)) == pytest.approx(FWDS)

    def test_check_forwards_passes_on_consistent_input(self):
        assert _surface().check_forwards() is True

    def test_check_forwards_flags_an_inconsistent_axis(self, caplog):
        s = FxVolInterpolation(VALDATE, list(EXPIRIES), [_smile(v) for v in PILLAR_VOLS],
                               [f * 1.05 for f in FWDS], **_market())
        with caplog.at_level('WARNING'):
            assert s.check_forwards() is False
        assert 'stored' in caplog.text

    def test_settlements_are_cached(self):
        s = _surface()
        for _ in range(5):
            s.forward(np.asarray(EXPIRIES, dtype=object))
        assert len(s._settle_cache) == len(EXPIRIES)

    def test_year_fraction_without_forward_raises(self):
        with pytest.raises(TypeError):
            _surface().vol_at_strike(1.0, 150.0)

    def test_year_fraction_with_explicit_forward_works(self):
        v = _surface().vol_at_strike(TIMES[1], FWDS[1], fwd=FWDS[1])
        assert float(v) == pytest.approx(PILLAR_VOLS[1][2])


class TestMoneynessDirection:
    def test_quoted_moneyness_is_reproduced_at_every_pillar(self):
        s = _surface()
        for i, e in enumerate(EXPIRIES):
            assert s.vol_at_moneyness(e, MONEYNESS) == pytest.approx(PILLAR_VOLS[i])

    def test_smile_keeps_its_shape_between_quoted_points(self):
        v = _surface().vol_at_moneyness(EXPIRIES[0], [-0.10, -0.03, 0.0, 0.03, 0.08])
        assert v[0] > v[1] > v[2]
        assert v[2] < v[3] < v[4]

    def test_moneyness_extrapolation_is_flat_outside_the_grid(self):
        s = _surface()
        assert float(s.vol_at_moneyness(EXPIRIES[0], -5.0)) == pytest.approx(PILLAR_VOLS[0][0])
        assert float(s.vol_at_moneyness(EXPIRIES[0], +5.0)) == pytest.approx(PILLAR_VOLS[0][-1])


class TestStrikeDirection:
    def test_strike_at_the_forward_is_the_atm_moneyness_vol(self):
        s = _surface()
        for i, e in enumerate(EXPIRIES):
            assert float(s.vol_at_strike(e, FWDS[i])) == pytest.approx(PILLAR_VOLS[i][2])

    def test_pillar_strikes_round_trip(self):
        s = _surface()
        for i, e in enumerate(EXPIRIES):
            assert s.vol_at_strike(e, s.pillar_strikes(i)) == pytest.approx(PILLAR_VOLS[i])

    def test_derived_and_explicit_forward_agree(self):
        s = _surface()
        k = np.linspace(0.75 * FWDS[1], 1.3 * FWDS[1], 25)
        assert s.vol_at_strike(EXPIRIES[1], k) == pytest.approx(s.vol_at_strike(EXPIRIES[1], k,
                                                                                fwd=FWDS[1]))

    def test_non_positive_strike_raises(self):
        with pytest.raises(ValueError):
            _surface().vol_at_strike(EXPIRIES[0], [100.0, 0.0])


class TestTimeDirection:
    @pytest.mark.parametrize('time_interp', ['var', 'vol2', 'vol'])
    def test_pillar_expiries_return_pillar_smiles(self, time_interp):
        s = _surface(time_interp=time_interp)
        for i, e in enumerate(EXPIRIES):
            assert s.vol_at_moneyness(e, MONEYNESS) == pytest.approx(PILLAR_VOLS[i])

    def test_var_mode_interpolates_linearly_in_total_variance(self):
        s = _surface(time_interp='var')
        m, t = -0.06, 0.5 * (TIMES[0] + TIMES[1])
        s0, s1 = _pillar_vol(0, m), _pillar_vol(1, m)
        w0, w1 = s0 ** 2 * TIMES[0], s1 ** 2 * TIMES[1]
        theta = (t - TIMES[0]) / (TIMES[1] - TIMES[0])
        assert float(s.vol_at_moneyness(t, m)) == pytest.approx(np.sqrt((w0 + theta * (w1 - w0)) / t))

    def test_vol2_mode_interpolates_linearly_in_variance(self):
        s = _surface(time_interp='vol2')
        m, t = -0.06, 0.5 * (TIMES[0] + TIMES[1])
        s0, s1 = _pillar_vol(0, m), _pillar_vol(1, m)
        theta = (t - TIMES[0]) / (TIMES[1] - TIMES[0])
        assert float(s.vol_at_moneyness(t, m)) == pytest.approx(
            np.sqrt(s0 ** 2 + theta * (s1 ** 2 - s0 ** 2)))

    def test_vol_mode_interpolates_linearly_in_vol(self):
        s = _surface(time_interp='vol')
        m, t = -0.06, 0.5 * (TIMES[0] + TIMES[1])
        s0, s1 = _pillar_vol(0, m), _pillar_vol(1, m)
        theta = (t - TIMES[0]) / (TIMES[1] - TIMES[0])
        assert float(s.vol_at_moneyness(t, m)) == pytest.approx(s0 + theta * (s1 - s0))

    @pytest.mark.parametrize('time_interp', ['var', 'vol2', 'vol'])
    def test_flat_term_structure_is_preserved(self, time_interp):
        """ For 'var' this is the sharp one: total variance must be exactly linear through 0 """
        s = _flat_surface(0.10, time_interp=time_interp)
        t = np.linspace(TIMES[0], TIMES[-1], 25)
        assert s.vol_at_moneyness(t, 0.0) == pytest.approx(0.10)

    def test_interpolated_vol_lies_between_the_pillar_vols(self):
        s = _surface()
        t = np.linspace(TIMES[0], TIMES[1], 20)[1:-1]
        v = s.vol_at_moneyness(t, 0.0)
        assert np.all(v > _pillar_vol(0, 0.0)) and np.all(v < _pillar_vol(1, 0.0))


class TestTimeExtrapolation:
    @pytest.mark.parametrize('time_extrap', ['flat', 'linear'])
    def test_before_first_pillar_is_always_flat_in_vol(self, time_extrap):
        s = _surface(time_extrap=time_extrap)
        short = np.asarray([0.001, 0.01, 0.05, 0.5 * TIMES[0]])
        assert s.vol_at_moneyness(short, 0.0) == pytest.approx(_pillar_vol(0, 0.0))

    @pytest.mark.parametrize('time_extrap', ['flat', 'linear'])
    def test_short_end_never_collapses_to_zero_vol(self, time_extrap):
        """ Regression: backward extrapolation of total variance used to go negative, get floored
            at zero and silently return a zero vol """
        s = _surface(time_extrap=time_extrap)
        assert np.all(s.vol_at_moneyness(np.asarray([1e-4, 1e-3, 1e-2]), 0.0) > 0.01)

    def test_after_last_pillar_is_flat_when_requested(self):
        s = _surface(time_extrap='flat')
        assert s.vol_at_moneyness(np.asarray([TIMES[-1] + 0.1, 5.0, 30.0]), 0.0) == pytest.approx(
            _pillar_vol(2, 0.0))

    def test_after_last_pillar_linear_follows_the_variance_line(self):
        s = _surface(time_extrap='linear', time_interp='var')
        m, t = 0.0, TIMES[-1] + 1.0
        s0, s1 = _pillar_vol(1, m), _pillar_vol(2, m)
        w0, w1 = s0 ** 2 * TIMES[-2], s1 ** 2 * TIMES[-1]
        theta = (t - TIMES[-2]) / (TIMES[-1] - TIMES[-2])
        assert float(s.vol_at_moneyness(t, m)) == pytest.approx(np.sqrt((w0 + theta * (w1 - w0)) / t))

    def test_linear_extrapolation_preserves_a_flat_term_structure(self):
        s = _flat_surface(0.10, time_extrap='linear')
        assert s.vol_at_moneyness(np.asarray([0.001, 5.0, 50.0]), 0.0) == pytest.approx(0.10)


class TestVectorization:
    def test_scalar_inputs_give_a_scalar_result(self):
        v = _surface().vol_at_moneyness(EXPIRIES[0], 0.0)
        assert np.shape(v) == ()

    def test_vector_shapes(self):
        s = _surface()
        assert s.vol_at_moneyness(EXPIRIES[0], MONEYNESS).shape == (len(MONEYNESS),)
        assert s.vol_at_moneyness(EXPIRIES, 0.0).shape == (len(EXPIRIES),)
        assert s.vol_at_moneyness(TIMES[:, None], MONEYNESS[None, :]).shape == (3, len(MONEYNESS))

    @pytest.mark.parametrize('time_extrap', ['flat', 'linear'])
    def test_vectorized_call_matches_the_scalar_loop(self, time_extrap):
        """ Points spread over every bracket and both extrapolation regions, so the per-pillar
            grouping in _smile_values has to dispatch each point to the right smile """
        s = _surface(time_extrap=time_extrap)
        t = np.asarray([0.01, 0.1, TIMES[0], 0.4, TIMES[1], 0.8, TIMES[2], 1.5, 9.0])
        m = np.asarray([-0.30, -0.12, -0.08, -0.02, 0.0, 0.03, 0.07, 0.10, 0.40])
        assert s.vol_at_moneyness(t, m) == pytest.approx(
            [float(s.vol_at_moneyness(ti, mi)) for ti, mi in zip(t, m)])

    def test_vol_grid_matches_pointwise_evaluation(self):
        s = _surface()
        k = np.asarray([130.0, 150.0, 170.0])
        g = s.vol_grid(EXPIRIES, k)
        assert g.shape == (len(EXPIRIES), len(k))
        for i, e in enumerate(EXPIRIES):
            assert g[i] == pytest.approx(s.vol_at_strike(e, k))


class TestVariance:
    def test_var_at_strike_is_vol_squared_times_time(self):
        s = _surface()
        k = np.asarray([140.0, 150.0, 160.0])
        assert s.var_at_strike(EXPIRIES[1], k) == pytest.approx(
            s.vol_at_strike(EXPIRIES[1], k) ** 2 * TIMES[1])

    def test_total_variance_increases_with_expiry_at_fixed_moneyness(self):
        s = _surface()
        t = np.linspace(TIMES[0], TIMES[-1], 40)
        assert np.all(np.diff(s.vol_at_moneyness(t, 0.0) ** 2 * t) > 0.0)


class TestCalendarCheck:
    def test_increasing_surface_passes(self):
        assert _surface().calendar_check() is True

    def test_flat_surface_passes(self):
        assert _flat_surface().calendar_check() is True

    def test_decreasing_total_variance_is_flagged(self, caplog):
        vols = [[0.30] * len(MONEYNESS), [0.05] * len(MONEYNESS), [0.05] * len(MONEYNESS)]
        s = FxVolInterpolation(VALDATE, list(EXPIRIES), [_smile(v) for v in vols], list(FWDS),
                               **_market())
        with caplog.at_level('WARNING'):
            assert s.calendar_check() is False
        assert 'Calendar arbitrage' in caplog.text

    def test_arbitrage_freedom_propagates_to_the_interpolated_surface(self):
        """ The whole point of interpolating variance at constant moneyness: clean pillars must
            give a clean surface everywhere in between, not just at the pillars. """
        s = _surface()
        assert s.calendar_check() is True
        t = np.linspace(TIMES[0], TIMES[-1], 300)
        m = np.linspace(-0.4, 0.4, 100)
        w = s.vol_at_moneyness(t[:, None], m[None, :]) ** 2 * t[:, None]
        assert np.all(np.diff(w, axis=0) >= -1e-15)


class TestDeltaQuotes:
    @staticmethod
    def _market_surface():
        data = CalibrationDataFileProvider().get_fxvol_data(PAIR, VALDATE)
        return data, interpolation_from_fxvol_data(data, MD_PROV, smile_interp='linear',
                                                   smile_extrap='flat')

    @pytest.mark.parametrize('tenor', ['1W', '1M', '6M', '1Y', '2Y'])
    def test_market_delta_quotes_round_trip(self, tenor):
        """ The calibrated 10d/25d put and call quotes must come back exactly """
        data, s = self._market_surface()
        r = [x for x in data['tenor_reports'] if x['tenor'] == tenor][0]
        expiry = dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
        for d, typ, idx in ((0.10, 'P', 0), (0.25, 'P', 1), (0.25, 'C', 3), (0.10, 'C', 4)):
            got = float(np.ravel(s.vol_at_delta(expiry, d, typ))[0])
            assert got == pytest.approx(r['vols'][idx], abs=1e-8)

    @pytest.mark.parametrize('tenor', ['1M', '1Y'])
    def test_strike_at_delta_recovers_the_calibrated_strikes(self, tenor):
        data, s = self._market_surface()
        r = [x for x in data['tenor_reports'] if x['tenor'] == tenor][0]
        expiry = dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
        for d, typ, idx in ((0.10, 'P', 0), (0.25, 'P', 1), (0.25, 'C', 3), (0.10, 'C', 4)):
            k = float(np.ravel(s.strike_at_delta(expiry, d, typ))[0])
            assert k == pytest.approx(r['strikes'][idx], rel=1e-8)

    def test_delta_quotes_are_vectorized_across_tenors(self):
        data, s = self._market_surface()
        reports = sorted(data['tenor_reports'], key=lambda r: r['expiry'])
        expiries = np.asarray([dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
                               for r in reports], dtype=object)
        got = s.vol_at_delta(expiries, 0.25, 'P')
        assert got == pytest.approx([r['vols'][1] for r in reports], abs=1e-8)

    def test_vol_at_delta_agrees_with_vol_at_strike_at_its_own_strike(self):
        data, s = self._market_surface()
        r = [x for x in data['tenor_reports'] if x['tenor'] == '1Y'][0]
        expiry = dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
        k = s.strike_at_delta(expiry, 0.25, 'C')
        assert s.vol_at_delta(expiry, 0.25, 'C') == pytest.approx(s.vol_at_strike(expiry, k))


class TestSinglePillar:
    def test_single_pillar_surface_is_flat_in_time(self):
        s = FxVolInterpolation(VALDATE, [EXPIRIES[1]], [_smile(PILLAR_VOLS[1])], [FWDS[1]],
                               **_market())
        t = np.asarray([0.01, 0.5, TIMES[1], 5.0])
        assert s.vol_at_moneyness(t, 0.0) == pytest.approx(PILLAR_VOLS[1][2])

    def test_single_pillar_surface_keeps_its_smile(self):
        s = FxVolInterpolation(VALDATE, [EXPIRIES[1]], [_smile(PILLAR_VOLS[1])], [FWDS[1]],
                               **_market())
        assert s.vol_at_moneyness(3.0, MONEYNESS) == pytest.approx(PILLAR_VOLS[1])


class TestFromCalibratedData:
    @staticmethod
    def _data():
        return CalibrationDataFileProvider().get_fxvol_data(PAIR, VALDATE)

    def test_builds_one_pillar_per_tenor_report(self):
        data = self._data()
        s = interpolation_from_fxvol_data(data, MD_PROV)
        assert len(s.expiries) == len(data['tenor_reports'])
        assert s.valdate == VALDATE and s.pair == PAIR

    def test_pillar_expiries_and_forwards_match_the_file(self):
        data = self._data()
        s = interpolation_from_fxvol_data(data, MD_PROV)
        reports = sorted(data['tenor_reports'], key=lambda r: r['expiry'])
        assert s.expiries == [dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
                              for r in reports]
        assert s.fwds == pytest.approx([r['fwd'] for r in reports])

    def test_stored_forwards_agree_with_the_curves(self):
        """ The axis was built with the calibration's forwards; the curves must reproduce them """
        assert interpolation_from_fxvol_data(self._data(), MD_PROV).check_forwards() is True

    def test_market_vols_are_reproduced_at_every_quoted_strike(self):
        data = self._data()
        s = interpolation_from_fxvol_data(data, MD_PROV, smile_interp='linear', smile_extrap='flat')
        for r in data['tenor_reports']:
            expiry = dt.datetime.strptime(r['expiry'], dts.DATE_FILE_FORMAT)
            assert s.vol_at_strike(expiry, r['strikes']) == pytest.approx(r['vols'])

    def test_missing_forward_raises(self):
        data = self._data()
        data['tenor_reports'][0].pop('fwd')
        with pytest.raises(KeyError):
            interpolation_from_fxvol_data(data, MD_PROV)

    def test_calendar_check_flags_the_1m_dip_in_the_test_data(self, caplog):
        """ The stored USDJPY set is mostly one repeated smile, with 1W and 1M carrying different
            quotes. The 1M smile sits below the block, so total variance falls from 3W to 1M on
            the call wing: a real (small) calendar arbitrage in the data, correctly detected. """
        s = interpolation_from_fxvol_data(self._data(), MD_PROV)
        with caplog.at_level('WARNING'):
            assert s.calendar_check() is False
        assert 'Calendar arbitrage' in caplog.text

    def test_calendar_check_passes_away_from_the_1m_pillar(self):
        data = self._data()
        data['tenor_reports'] = [r for r in data['tenor_reports'] if r['tenor'] != '1M']
        assert interpolation_from_fxvol_data(data, MD_PROV).calendar_check() is True

    def test_whole_market_surface_evaluates_in_one_vectorized_call(self):
        s = interpolation_from_fxvol_data(self._data(), MD_PROV)
        t = np.linspace(0.01, 5.0, 60)
        m = np.linspace(-0.5, 0.5, 40)
        v = s.vol_at_moneyness(t[:, None], m[None, :])
        assert v.shape == (60, 40)
        assert np.all(np.isfinite(v)) and np.all(v > 0.0)
