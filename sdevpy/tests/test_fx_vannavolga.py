import pytest
import numpy as np
from itertools import pairwise
from sdevpy.analytics import black
from sdevpy.volatility.fx.fx_vannavolga import (VannaVolgaSmile, smile_from_quotes, vv_weights,
                                                lagrange_weights, atm_dns_strike, bs_vega)
from sdevpy.volatility.fx.fx_smilecalib import market_strangle, calibrate_smile_strangle

SPOT, R_D, R_F, EXPIRY = 1.10, 0.04, 0.02, 1.0
ATM_VOL, RR, BF = 0.10, -0.01, 0.0025


def _smile(**kwargs):
    params = dict(spot=SPOT, r_d=R_D, r_f=R_F, expiry=EXPIRY, atm_vol=ATM_VOL, rr=RR, bf=BF)
    params.update(kwargs)
    return smile_from_quotes(**params)


def _vv_build_smile(rr=RR, prem_adjusted=False):
    """ build_smile for calibrate_smile_strangle: rebuilds a VannaVolgaSmile from a candidate
        smile strangle. `rr` must match whatever rr is passed to calibrate_smile_strangle at
        the same call site -- the objective function needs both to describe the same skew. """
    def build_smile(bf):
        return _smile(rr=rr, bf=bf, prem_adjusted=prem_adjusted, extrapolation='none')
    return build_smile


class TestPillarConstruction:
    def test_quote_triple_maps_to_pillar_vols(self):
        s = _smile()
        assert s.vol_call == pytest.approx(ATM_VOL + BF + 0.5 * RR)
        assert s.vol_put == pytest.approx(ATM_VOL + BF - 0.5 * RR)

    def test_pillar_strikes_are_increasing(self):
        s = _smile()
        assert s.k_put < s.k_atm < s.k_call

    def test_negative_rr_puts_skew_on_the_put_wing(self):
        assert _smile().vol_put > _smile().vol_call

    def test_atm_dns_strike_above_forward_when_not_prem_adjusted(self):
        assert atm_dns_strike(1.12, 0.10, 1.0, prem_adjusted=False) > 1.12

    def test_atm_dns_strike_below_forward_when_prem_adjusted(self):
        assert atm_dns_strike(1.12, 0.10, 1.0, prem_adjusted=True) < 1.12

    def test_non_increasing_pillars_raise(self):
        with pytest.raises(ValueError):
            VannaVolgaSmile(fwd=1.12, expiry=1.0, k_put=1.2, k_atm=1.1, k_call=1.0,
                            vol_put=0.11, atm_vol=0.10, vol_call=0.098)

    def test_bad_extrapolation_mode_raises(self):
        with pytest.raises(ValueError):
            _smile(extrapolation='quadratic')


class TestWeights:
    def test_weights_are_kronecker_delta_at_pillars(self):
        s = _smile()
        for i, k in enumerate([s.k_put, s.k_atm, s.k_call]):
            w = np.asarray(vv_weights(k, s.k_put, s.k_atm, s.k_call,
                                      s.fwd, s.expiry, s.atm_vol)).ravel()
            expected = np.zeros(3)
            expected[i] = 1.0
            assert np.allclose(w, expected, atol=1e-12)

    def test_lagrange_weights_sum_to_one(self):
        s = _smile()
        w1, w2, w3 = lagrange_weights(np.linspace(0.90, 1.40, 21), s.k_put, s.k_atm, s.k_call)
        assert np.allclose(w1 + w2 + w3, 1.0, atol=1e-12)

    def test_vega_is_positive_and_peaks_near_the_forward(self):
        strikes = np.linspace(0.90, 1.40, 51)
        vega = bs_vega(1.12, strikes, 1.0, 0.10)
        assert np.all(vega > 0.0)
        assert strikes[np.argmax(vega)] == pytest.approx(1.12, abs=0.05)


class TestExactSmile:
    def test_reprices_pillars_exactly(self):
        s = _smile(extrapolation='none')
        strikes = np.array([s.k_put, s.k_atm, s.k_call])
        assert np.allclose(s.vol(strikes),
                           np.array([s.vol_put, s.atm_vol, s.vol_call]), atol=1e-10)

    def test_flat_input_smile_stays_flat(self):
        s = _smile(rr=0.0, bf=0.0, extrapolation='none')
        assert np.allclose(s.vol(np.linspace(0.85, 1.55, 15)), ATM_VOL, atol=1e-10)

    def test_call_and_put_prices_satisfy_forward_parity(self):
        s = _smile(extrapolation='none')
        strikes = np.linspace(0.95, 1.35, 9)
        assert np.allclose(s.price(strikes, True) - s.price(strikes, False),
                           s.fwd - strikes, atol=1e-10)

    def test_scalar_and_array_queries_agree(self):
        s = _smile()
        assert float(s.vol(1.15)) == pytest.approx(float(np.atleast_1d(s.vol(np.array([1.15])))[0]))


class TestFirstOrder:
    def test_first_order_also_reprices_pillars(self):
        s = _smile(extrapolation='none')
        strikes = np.array([s.k_put, s.k_atm, s.k_call])
        assert np.allclose(s.vol(strikes, method='first_order'),
                           np.array([s.vol_put, s.atm_vol, s.vol_call]), atol=1e-12)

    def test_first_order_tracks_exact_inside_the_quoted_range(self):
        s = _smile(extrapolation='none')
        strikes = np.linspace(s.k_put, s.k_call, 11)
        assert np.max(np.abs(s.vol(strikes) - s.vol(strikes, method='first_order'))) < 1e-4

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _smile().vol(1.15, method='second_order')


class TestExtrapolation:
    def test_flat_extrapolation_below_put_pillar(self):
        s = _smile(extrapolation='flat')
        assert float(s.vol(0.80)) == pytest.approx(s.vol_put)

    def test_flat_extrapolation_above_call_pillar(self):
        s = _smile(extrapolation='flat')
        assert float(s.vol(1.80)) == pytest.approx(s.vol_call)

    def test_flat_extrapolation_leaves_interior_untouched(self):
        flat, raw = _smile(extrapolation='flat'), _smile(extrapolation='none')
        strikes = np.linspace(flat.k_put * 1.01, flat.k_call * 0.99, 9)
        assert np.allclose(flat.vol(strikes), raw.vol(strikes), atol=1e-12)

    def test_raw_extrapolation_diverges_from_first_order_in_the_wings(self):
        # Why 'flat' is the default: outside the pillars neither construction is pinned by
        # market information, and they stop agreeing.
        s = _smile(extrapolation='none')
        assert abs(float(s.vol(0.95)) - float(s.vol(0.95, method='first_order'))) > 1e-3


class TestMarketStrangleCalibration:
    def test_zero_risk_reversal_leaves_butterfly_unchanged(self):
        # With no skew the pillar strikes coincide with the market-strangle strikes, so the
        # calibration is a provable no-op -- the strongest check on the routine.
        bf = calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=0.0, ms=0.0025,
                                      build_smile=_vv_build_smile(rr=0.0))
        assert bf == pytest.approx(0.0025, abs=1e-11)

    def test_calibrated_smile_reprices_the_market_strangle(self):
        s = _smile(rr=-0.01, bf=0.0025, market_strangle_quote=True, extrapolation='none')
        k_put, k_call, target, _ = market_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, 0.0025)
        priced = (black.price(EXPIRY, k_call, True, s.fwd, float(s.vol(k_call)))
                  + black.price(EXPIRY, k_put, False, s.fwd, float(s.vol(k_put))))
        assert priced == pytest.approx(target, abs=1e-10)

    def test_market_strangle_strikes_straddle_the_forward(self):
        k_put, k_call, price, vol_ms = market_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, 0.0025)
        fwd = SPOT * np.exp((R_D - R_F) * EXPIRY)
        assert k_put < fwd < k_call
        assert price > 0.0
        assert vol_ms == pytest.approx(ATM_VOL + 0.0025)

    def test_smile_butterfly_exceeds_market_butterfly_when_skewed(self):
        assert calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=-0.02, ms=0.0025,
                                        build_smile=_vv_build_smile(rr=-0.02)) > 0.0025

    def test_correction_grows_with_skew(self):
        gaps = [calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=r, ms=0.0025,
                                         build_smile=_vv_build_smile(rr=r))
                for r in (-0.005, -0.01, -0.02, -0.04)]
        assert all(b > a for a, b in pairwise(gaps))

    def test_flag_off_uses_the_quote_directly(self):
        s = _smile(rr=-0.02, bf=0.0025, market_strangle_quote=False)
        assert s.smile_butterfly == pytest.approx(0.0025)
        assert s.market_butterfly is None

    def test_flag_on_records_both_butterflies(self):
        s = _smile(rr=-0.02, bf=0.0025, market_strangle_quote=True)
        assert s.market_butterfly == pytest.approx(0.0025)
        assert s.smile_butterfly > 0.0025

    def test_correction_is_not_symmetric_in_the_skew(self):
        # The DNS ATM strike sits above the forward by 0.5*sigma^2*T, so the three-strike grid
        # is not symmetric in log-moneyness and flipping the skew does not mirror it. The gap
        # is ~4% at T=1Y and grows with sigma^2*T (2% at 3M, 6% at 2Y).
        pos = calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=0.02, ms=0.0025,
                                       build_smile=_vv_build_smile(rr=0.02))
        neg = calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=-0.02, ms=0.0025,
                                       build_smile=_vv_build_smile(rr=-0.02))
        assert pos != pytest.approx(neg, abs=1e-6)
        assert abs(pos - neg) / neg == pytest.approx(0.044, abs=0.005)


class TestPremiumAdjusted:
    def test_prem_adjusted_smile_reprices_pillars(self):
        # Guards the double-root trap: the PA call delta is non-monotonic, and the deep-ITM
        # root (0.286 vs a 1.122 forward) would give non-monotonic pillars.
        s = smile_from_quotes(SPOT, R_D, R_F, EXPIRY, ATM_VOL, RR, BF,
                              prem_adjusted=True, extrapolation='none')
        strikes = np.array([s.k_put, s.k_atm, s.k_call])
        assert np.allclose(s.vol(strikes),
                           np.array([s.vol_put, s.atm_vol, s.vol_call]), atol=1e-10)

    def test_prem_adjusted_call_pillar_is_the_otm_root(self):
        s = smile_from_quotes(SPOT, R_D, R_F, EXPIRY, ATM_VOL, RR, BF, prem_adjusted=True)
        assert s.k_call > s.fwd

    def test_prem_adjusted_market_strangle_calibration(self):
        bf = calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=-0.01, ms=0.0025,
                                      build_smile=_vv_build_smile(rr=-0.01, prem_adjusted=True),
                                       prem_adjusted=True)
        assert bf == pytest.approx(0.00284682, abs=1e-7)


class TestRegression:
    """ Reference values from an independent implementation (erf-based normal CDF, closed-form
        plain-delta strike inversion), cross-checked against the machine-precision structural
        properties above. Re-derive independently if the model changes -- do not rebase. """

    def test_pillar_strikes(self):
        s = _smile()
        assert s.k_put == pytest.approx(1.05156560, abs=1e-6)
        assert s.k_atm == pytest.approx(1.12784663, abs=1e-6)
        assert s.k_call == pytest.approx(1.20235808, abs=1e-6)

    def test_smile_values(self):
        s = _smile(extrapolation='none')
        strikes = np.array([1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
        expected = np.array([0.11487651, 0.10770597, 0.10212950, 0.09879267,
                             0.09751381, 0.09806641, 0.10026222])
        assert np.allclose(s.vol(strikes), expected, atol=1e-7)

    @pytest.mark.parametrize("rr, ms, expected", [
        (-0.005, 0.0025, 0.00252780),
        (-0.010, 0.0025, 0.00265021),
        (-0.020, 0.0025, 0.00317565),
        (-0.040, 0.0025, 0.00528512),
        (-0.010, 0.0050, 0.00510236),
    ])
    def test_calibrated_butterfly(self, rr, ms, expected):
        bf = calibrate_smile_strangle(SPOT, R_D, R_F, EXPIRY, ATM_VOL, rr=rr, ms=ms,
                                      build_smile=_vv_build_smile(rr=rr))
        assert bf == pytest.approx(expected, abs=1e-7)


################### TEMP ###############################
import datetime as dt
import numpy as np
import pytest
from sdevpy.market import fxspot
from sdevpy.market.fxvolsurface import fxvolsurfacedata_from_file, wingvols_from_butterfly
from sdevpy.utilities import dates as dts
from sdevpy.utilities import timegrids
from sdevpy.market.fileprovider import MarketDataFileProvider
from sdevpy.volatility.fx import fx_vannavolga
from sdevpy.volatility.fx.fx_deltastrike import strike_from_delta

PAIR = "USDJPY"
VALDATE = dt.datetime(2025, 12, 15)
VIEW_EXPIRY_IDX = 0


def _run_pipeline():
    """ Mirrors ex_fx_vol_market.py end to end, returning every intermediate value the
        regression checks pin against. Kept as a single function so a refactor of the script
        can be checked by re-running this, not by re-deriving expected values by hand. """
    provider = MarketDataFileProvider()
    ccy1, ccy2 = fxspot.parse_fx_pair(PAIR)
    forccy, domccy = fxspot.conventional_pair_name(ccy1, ccy2)

    file = provider.root / "fxoptions" / PAIR / (VALDATE.strftime(dts.DATE_FILE_FORMAT) + ".json")
    data = fxvolsurfacedata_from_file(file)

    expiry = data.expiries[VIEW_EXPIRY_IDX]
    atm_vol = data.atm_vols[VIEW_EXPIRY_IDX]
    deltas = data.deltas[VIEW_EXPIRY_IDX]
    rr = data.rr[VIEW_EXPIRY_IDX]
    bf = data.bf[VIEW_EXPIRY_IDX]

    spot = provider.get_fx_spot(forccy, domccy, VALDATE)
    t = timegrids.model_time(VALDATE, expiry)

    forcurve = provider.get_xccycurve(forccy, VALDATE)
    domcurve = provider.get_xccycurve(domccy, VALDATE)
    df_for = forcurve.discount(expiry)
    df_dom = domcurve.discount(expiry)
    r_for, r_dom = -np.log(df_for) / t, -np.log(df_dom) / t
    fwd = spot * df_for / df_dom

    market_strikes, market_vols = [], []
    for d, r, b in zip(deltas, rr, bf, strict=True):
        if data.market_strangle_quote:
            vol_put, vol_call = fx_vannavolga.wingvols_from_market_strangle_vv(
                spot, r_dom, r_for, t, atm_vol, r, b, delta=d)
        else:
            vol_put, vol_call = wingvols_from_butterfly(atm_vol, r, b)
        k_put = float(strike_from_delta(spot, df_for, df_dom, t, vol_put, -d, 'P').k)
        k_call = float(strike_from_delta(spot, df_for, df_dom, t, vol_call, d, 'C').k)
        # k_put = float(strike_from_delta(spot, r_dom, r_for, t, vol_put, -d, 'P').k)
        # k_call = float(strike_from_delta(spot, r_dom, r_for, t, vol_call, d, 'C').k)
        market_strikes += [k_put, k_call]
        market_vols += [vol_put, vol_call]

    k_atm = fx_vannavolga.atm_dns_strike(fwd, atm_vol, t)
    market_strikes.append(k_atm)
    market_vols.append(atm_vol)

    delta_idx = 0
    s = fx_vannavolga.smile_from_quotes(spot=spot, r_d=r_dom, r_f=r_for, expiry=t, atm_vol=atm_vol,
                                        rr=rr[delta_idx], bf=bf[delta_idx], delta=deltas[delta_idx])

    return {
        'forccy': forccy, 'domccy': domccy, 'spot': spot, 't': t,
        'df_for': df_for, 'df_dom': df_dom, 'r_for': r_for, 'r_dom': r_dom, 'fwd': fwd,
        'market_strikes': market_strikes, 'market_vols': market_vols, 'smile': s,
    }


class TestExFxVolMarketRegression:
    """ Reference values captured by running this exact pipeline against the real
        sdevpy/tests/data/marketdata/fxoptions/USDJPY file, expiry index 0 (20-Jan-2026,
        the only section in that file with realistic rr/bf -- the others are known
        placeholder data). Re-derive by re-running _run_pipeline() if the market data file,
        or any function in the pipeline, deliberately changes -- do not rebase blindly. """

    def test_pair_and_market_data(self):
        r = _run_pipeline()
        assert (r['forccy'], r['domccy']) == ('USD', 'JPY')
        assert r['spot'] == pytest.approx(150.0)
        assert r['t'] == pytest.approx(0.09863013698630137, abs=1e-12)

    def test_rates_and_forward(self):
        r = _run_pipeline()
        assert r['df_for'] == pytest.approx(0.9995546727625232, abs=1e-12)
        assert r['df_dom'] == pytest.approx(0.9983732377760548, abs=1e-12)
        assert r['r_for'] == pytest.approx(0.004516129032258047, abs=1e-9)
        assert r['r_dom'] == pytest.approx(0.016506991555613408, abs=1e-9)
        assert r['fwd'] == pytest.approx(150.17750400477985, abs=1e-6)

    def test_market_points(self):
        r = _run_pipeline()
        expected_strikes = [146.876691576927, 153.3875242928018,
                            143.59489513179864, 155.9674117567631, 150.2515824081475]
        expected_vols = [0.10767373557813655, 0.09767373557813654,
                         0.11295060923945906, 0.09295060923945907, 0.1]
        assert np.allclose(r['market_strikes'], expected_strikes, atol=1e-6)
        assert np.allclose(r['market_vols'], expected_vols, atol=1e-9)

    def test_interpolation_smile_pillars(self):
        s = _run_pipeline()['smile']
        assert (s.k_put, s.k_atm, s.k_call) == pytest.approx(
            (146.8818234068457, 150.2515824081475, 153.38162590837575), abs=1e-6)
        assert (s.vol_put, s.atm_vol, s.vol_call) == pytest.approx((0.1075, 0.1, 0.0975), abs=1e-9)

    def test_interpolation_sample_values(self):
        s = _run_pipeline()['smile']
        strikes = [140.0, 145.0, s.fwd, 150.0, 155.0, 160.0]
        expected = [0.10750000000000001, 0.10750000000000001, 0.10011136451921224,
                   0.10038831753050967, 0.0975, 0.0975]
        got = [float(s.vol(k)) for k in strikes]
        assert np.allclose(got, expected, atol=1e-9)
