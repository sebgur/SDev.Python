import pytest
import numpy as np
from sdevpy.analytics import black
from sdevpy.volatility.impliedvol.optionsurface import (
    OptionQuoteType, OptionTarget, calibration_targets,
    keep_positive, check_expiries_and_forwards,
    check_model, convert_option, convert_target, convert_to_target_values,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_target(expiry=1.0, forward=0.04, strike=0.04, market_input=0.20,
                market_shift=0.0, is_call=True,
                quote_type=OptionQuoteType.LogNormalVol, is_atm=True):
    return OptionTarget(expiry=expiry, forward=forward, strike=strike,
                        market_input=market_input, market_shift=market_shift,
                        is_call=is_call, quote_type=quote_type, is_atm=is_atm)


# ── OptionTarget dataclass ────────────────────────────────────────────────────

class TestOptionTarget:
    def test_defaults(self):
        t = OptionTarget(expiry=1.0, forward=0.04, strike=0.04, market_input=0.20, market_shift=0.0)
        assert t.is_call is True
        assert t.quote_type == OptionQuoteType.LogNormalVol
        assert t.is_atm is True

    def test_custom_values_stored(self):
        t = make_target(expiry=2.0, forward=0.05, strike=0.06,
                        quote_type=OptionQuoteType.NormalVol, is_atm=False)
        assert t.expiry == 2.0 and t.forward == 0.05 and t.strike == 0.06
        assert t.quote_type == OptionQuoteType.NormalVol
        assert t.is_atm is False


# ── calibration_targets ───────────────────────────────────────────────────────

class TestCalibrationTargets:
    _exp = [0.5, 1.0]
    _fwd = [0.04, 0.04]
    _str = [[0.03, 0.04, 0.05], [0.03, 0.04, 0.05]]
    _vol = [[0.20, 0.20, 0.20], [0.20, 0.20, 0.20]]

    def test_returns_one_entry_per_expiry(self):
        prices, ftols = calibration_targets(self._exp, self._fwd, self._str, self._vol)
        assert len(prices) == 2 and len(ftols) == 2

    def test_ftols_are_positive(self):
        _, ftols = calibration_targets(self._exp, self._fwd, self._str, self._vol)
        assert all(f > 0 for f in ftols)

    def test_call_prices_non_negative(self):
        prices, _ = calibration_targets(self._exp, self._fwd, self._str, self._vol,
                                        option_type='call')
        assert all(np.all(np.asarray(p) >= 0) for p in prices)

    def test_put_prices_non_negative(self):
        prices, _ = calibration_targets(self._exp, self._fwd, self._str, self._vol,
                                        option_type='put')
        assert all(np.all(np.asarray(p) >= 0) for p in prices)

    def test_straddle_exceeds_call_at_each_point(self):
        p_call, _ = calibration_targets(self._exp, self._fwd, self._str, self._vol,
                                        option_type='call')
        p_straddle, _ = calibration_targets(self._exp, self._fwd, self._str, self._vol,
                                            option_type='straddle')
        for pc, ps in zip(p_call, p_straddle):
            assert np.all(np.asarray(ps) >= np.asarray(pc))

    def test_unknown_type_falls_through_to_straddle(self):
        p_straddle, _ = calibration_targets(self._exp, self._fwd, self._str, self._vol,
                                            option_type='straddle')
        p_other, _ = calibration_targets(self._exp, self._fwd, self._str, self._vol,
                                         option_type='whatever')
        for ps, po in zip(p_straddle, p_other):
            np.testing.assert_array_almost_equal(ps, po)


# ── keep_positive ─────────────────────────────────────────────────────────────

class TestKeepPositive:
    def test_keeps_all_when_all_positive(self):
        targets = [[make_target(forward=0.04, strike=0.03),
                    make_target(forward=0.04, strike=0.05)]]
        assert len(keep_positive(targets)[0]) == 2

    def test_removes_negative_forward(self):
        targets = [[make_target(forward=-0.01, strike=0.03),
                    make_target(forward=0.04, strike=0.03)]]
        result = keep_positive(targets)
        assert len(result[0]) == 1
        assert result[0][0].forward == 0.04

    def test_removes_negative_strike(self):
        targets = [[make_target(forward=0.04, strike=-0.01),
                    make_target(forward=0.04, strike=0.03)]]
        result = keep_positive(targets)
        assert len(result[0]) == 1
        assert result[0][0].strike == 0.03

    def test_removes_zero_forward(self):
        targets = [[make_target(forward=0.0, strike=0.03)]]
        assert keep_positive(targets) == []

    def test_empty_expiry_silently_dropped_by_default(self):
        targets = [[make_target(forward=-0.01, strike=-0.01)],
                   [make_target(forward=0.04, strike=0.03)]]
        result = keep_positive(targets)
        assert len(result) == 1

    def test_raises_on_empty_expiry_when_flag_false(self):
        targets = [[make_target(forward=-0.01, strike=-0.01)]]
        with pytest.raises(ValueError, match="No positive-rate"):
            keep_positive(targets, skip_empty_expiries=False)


# ── check_expiries_and_forwards ───────────────────────────────────────────────

class TestCheckExpiriesAndForwards:
    def test_consistent_targets_pass(self):
        targets = [[make_target(expiry=1.0, forward=0.04),
                    make_target(expiry=1.0, forward=0.04)]]
        check_expiries_and_forwards(targets)  # no raise

    def test_multiple_expiry_sections_pass(self):
        targets = [
            [make_target(expiry=0.5, forward=0.03), make_target(expiry=0.5, forward=0.03)],
            [make_target(expiry=1.0, forward=0.04), make_target(expiry=1.0, forward=0.04)],
        ]
        check_expiries_and_forwards(targets)  # no raise

    def test_raises_on_empty_expiry_section(self):
        with pytest.raises(ValueError, match="No options found"):
            check_expiries_and_forwards([[]])

    def test_raises_on_inconsistent_expiry(self):
        targets = [[make_target(expiry=1.0, forward=0.04),
                    make_target(expiry=2.0, forward=0.04)]]
        with pytest.raises(ValueError, match="Inconsistent expiries"):
            check_expiries_and_forwards(targets)

    def test_raises_on_inconsistent_forward(self):
        targets = [[make_target(expiry=1.0, forward=0.04),
                    make_target(expiry=1.0, forward=0.05)]]
        with pytest.raises(ValueError, match="Inconsistent expiries"):
            check_expiries_and_forwards(targets)


# ── check_model ───────────────────────────────────────────────────────────────

class TestCheckModel:
    def test_lognormal_positive_inputs_pass(self):
        check_model(0.04, 0.03, OptionQuoteType.LogNormalVol, 0.0)

    def test_lognormal_negative_fwd_raises(self):
        with pytest.raises(ValueError, match="Negative forward"):
            check_model(-0.01, 0.03, OptionQuoteType.LogNormalVol, 0.0)

    def test_lognormal_negative_strike_raises(self):
        with pytest.raises(ValueError, match="Negative strike"):
            check_model(0.04, -0.01, OptionQuoteType.LogNormalVol, 0.0)

    def test_shifted_lognormal_shift_rescues_negative_rate(self):
        # fwd=-0.02 + shift=0.03 = 0.01 > 0 → should not raise
        check_model(-0.02, 0.01, OptionQuoteType.ShiftedLogNormalVol, 0.03)

    def test_shifted_lognormal_negative_shifted_strike_raises(self):
        # strike=-0.02 + shift=0.005 = -0.015 < 0 → raises
        with pytest.raises(ValueError, match="Negative strike"):
            check_model(0.01, -0.02, OptionQuoteType.ShiftedLogNormalVol, 0.005)

    def test_normal_vol_allows_negative_rates(self):
        check_model(-0.01, -0.01, OptionQuoteType.NormalVol, 0.0)  # no raise

    def test_forward_premium_allows_negative_rates(self):
        check_model(-0.01, -0.01, OptionQuoteType.ForwardPremium, 0.0)  # no raise


# ── convert_option ────────────────────────────────────────────────────────────

class TestConvertOption:
    T, K, F = 1.0, 0.04, 0.04
    LN_VOL = 0.20

    def test_same_type_and_shift_returns_input_unchanged(self):
        result = convert_option(self.T, self.K, True, self.F,
                                self.LN_VOL, OptionQuoteType.LogNormalVol, 0.0,
                                OptionQuoteType.LogNormalVol, 0.0)
        assert result == self.LN_VOL

    def test_lognormal_to_forward_premium_positive(self):
        result = convert_option(self.T, self.K, True, self.F,
                                self.LN_VOL, OptionQuoteType.LogNormalVol, 0.0,
                                OptionQuoteType.ForwardPremium, 0.0)
        assert result > 0.0

    def test_lognormal_to_forward_premium_roundtrip(self):
        fwd_prem = convert_option(self.T, self.K, True, self.F,
                                  self.LN_VOL, OptionQuoteType.LogNormalVol, 0.0,
                                  OptionQuoteType.ForwardPremium, 0.0)
        recovered = convert_option(self.T, self.K, True, self.F,
                                   fwd_prem, OptionQuoteType.ForwardPremium, 0.0,
                                   OptionQuoteType.LogNormalVol, 0.0)
        assert recovered == pytest.approx(self.LN_VOL, rel=1e-6)

    def test_lognormal_to_normal_vol_roundtrip(self):
        nvol = convert_option(self.T, self.K, True, self.F,
                              self.LN_VOL, OptionQuoteType.LogNormalVol, 0.0,
                              OptionQuoteType.NormalVol, 0.0)
        recovered = convert_option(self.T, self.K, True, self.F,
                                   nvol, OptionQuoteType.NormalVol, 0.0,
                                   OptionQuoteType.LogNormalVol, 0.0)
        assert recovered == pytest.approx(self.LN_VOL, rel=1e-4)

    def test_forward_premium_to_normal_vol(self):
        fwd_prem = black.price(self.T, self.K, True, self.F, self.LN_VOL)
        result = convert_option(self.T, self.K, True, self.F,
                                fwd_prem, OptionQuoteType.ForwardPremium, 0.0,
                                OptionQuoteType.NormalVol, 0.0)
        assert result > 0.0

    def test_shifted_lognormal_to_forward_premium_and_back(self):
        shift = 0.03
        fwd, strike = 0.01, 0.01
        fwd_prem = convert_option(self.T, strike, True, fwd,
                                  self.LN_VOL, OptionQuoteType.ShiftedLogNormalVol, shift,
                                  OptionQuoteType.ForwardPremium, 0.0)
        recovered = convert_option(self.T, strike, True, fwd,
                                   fwd_prem, OptionQuoteType.ForwardPremium, 0.0,
                                   OptionQuoteType.ShiftedLogNormalVol, shift)
        assert recovered == pytest.approx(self.LN_VOL, rel=1e-4)

    def test_same_type_different_shift_does_convert(self):
        # shift1 ≠ shift2 → must convert, not pass through
        shift1, shift2 = 0.02, 0.03
        fwd, strike = 0.01, 0.01
        result = convert_option(self.T, strike, True, fwd,
                                self.LN_VOL, OptionQuoteType.ShiftedLogNormalVol, shift1,
                                OptionQuoteType.ShiftedLogNormalVol, shift2)
        assert result != self.LN_VOL


# ── convert_target ────────────────────────────────────────────────────────────

class TestConvertTarget:
    def test_preserves_expiry_forward_strike(self):
        src = make_target(expiry=2.0, forward=0.04, strike=0.05, market_input=0.20)
        result = convert_target(src, OptionQuoteType.ForwardPremium, 0.0)
        assert result.expiry == 2.0
        assert result.forward == 0.04
        assert result.strike == 0.05

    def test_sets_new_quote_type_and_shift(self):
        src = make_target(quote_type=OptionQuoteType.LogNormalVol, market_input=0.20)
        result = convert_target(src, OptionQuoteType.NormalVol, 0.0)
        assert result.quote_type == OptionQuoteType.NormalVol
        assert result.market_shift == 0.0

    def test_preserves_is_call_and_is_atm(self):
        src = make_target(is_call=False, is_atm=False)
        result = convert_target(src, OptionQuoteType.ForwardPremium, 0.0)
        assert result.is_call is False
        assert result.is_atm is False

    def test_market_input_changes_after_conversion(self):
        src = make_target(market_input=0.20, quote_type=OptionQuoteType.LogNormalVol)
        result = convert_target(src, OptionQuoteType.NormalVol, 0.0)
        assert result.market_input != src.market_input


# ── convert_to_target_values ──────────────────────────────────────────────────

class TestConvertToTargetValues:
    def test_successful_conversion_preserves_structure(self):
        targets = [[make_target(market_input=0.20), make_target(market_input=0.25)]]
        result = convert_to_target_values(targets, OptionQuoteType.NormalVol, 0.0)
        assert len(result) == 1 and len(result[0]) == 2
        assert all(t.quote_type == OptionQuoteType.NormalVol for t in result[0])

    def test_multiple_expiries_preserved(self):
        targets = [
            [make_target(expiry=0.5, market_input=0.20)],
            [make_target(expiry=1.0, market_input=0.25)],
        ]
        result = convert_to_target_values(targets, OptionQuoteType.ForwardPremium, 0.0)
        assert len(result) == 2

    def test_failed_target_is_silently_skipped(self):
        # Negative fwd with LogNormalVol fails check_model → silently dropped
        bad = make_target(forward=-0.01, strike=0.03, quote_type=OptionQuoteType.LogNormalVol)
        good = make_target(forward=0.04, strike=0.03, market_input=0.20)
        result = convert_to_target_values([[bad, good]], OptionQuoteType.ForwardPremium, 0.0)
        assert len(result) == 1 and len(result[0]) == 1

    def test_expiry_dropped_when_all_targets_fail(self):
        bad = make_target(forward=-0.01, strike=0.03, quote_type=OptionQuoteType.LogNormalVol)
        result = convert_to_target_values([[bad]], OptionQuoteType.ForwardPremium, 0.0)
        assert result == []
