import pytest
import datetime as dt
import numpy as np
from sdevpy.market import fxspot
from sdevpy.market.fxforward import FxForwardCurve, fx_spot_lag, fx_spot_date


def test_parse_fx_pair():
    assert fxspot.parse_fx_pair('EURUSD') == ('EUR', 'USD')

    # with pytest.raises(ValueError):
    #     fxspot.parse_fx_pair('EURO')


def test_use_conventional_pair():
    # EUR is in USD_IS_QUOTE -> EUR/USD
    assert fxspot.usd_conventional_pair('EUR') == ('EUR', 'USD')
    # JPY is not in USD_IS_QUOTE -> USD/JPY
    assert fxspot.usd_conventional_pair('JPY') == ('USD', 'JPY')


def test_is_inverted_quote():
    assert fxspot.is_inverted_quote('EUR') is False
    assert fxspot.is_inverted_quote('JPY') is True

    # is_inverted_quote should always agree with usd_conventional_pair's own answer
    for ccy in ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'MXN', 'ZAR']:
        forccy, _ = fxspot.usd_conventional_pair(ccy)
        assert fxspot.is_inverted_quote(ccy) == (forccy == 'USD')


def test_fx_spot_lag():
    assert fx_spot_lag('EUR', 'USD') == 2
    assert fx_spot_lag('USD', 'CAD') == 1
    assert fx_spot_lag('USD', 'CAD') == fx_spot_lag('CAD', 'USD')


class TestFxSpotDate:
    VALDATE = dt.datetime(2025, 8, 12)  # a Tuesday

    def test_same_currency_raises(self):
        with pytest.raises(ValueError):
            fx_spot_date(self.VALDATE, 'EUR', 'EUR')

    def test_usd_pair_two_day_lag(self):
        # Tue -> Thu, no weekend/holiday in the way
        assert fx_spot_date(self.VALDATE, 'EUR', 'USD') == dt.datetime(2025, 8, 14)

    def test_usd_cad_one_day_lag(self):
        print(fx_spot_date(self.VALDATE, 'USD', 'CAD'))
        assert fx_spot_date(self.VALDATE, 'USD', 'CAD') == dt.datetime(2025, 8, 13)

    def test_cross_pair_matches_usd_pair_lag(self):
        # Neither leg is USD, but same total lag as a direct USD pair with no holidays in play
        assert fx_spot_date(self.VALDATE, 'EUR', 'GBP') == dt.datetime(2025, 8, 14)


class ConstDiscountCurve:
    """ Test double: flat continuously-compounded discount curve """
    def __init__(self, valdate, rate):
        self.valdate = valdate
        self.rate = rate

    def discount(self, date):
        t = (date - self.valdate).days / 365.0
        return np.exp(-self.rate * t)

    def discount_float(self, t):
        return np.exp(-self.rate * t)


class TestFxForwardCurveMath:
    """ Isolates the CIRP math in load_calibrated/value from spot_date()'s calendar logic
        by monkeypatching spot_date() directly -- these pass independently of the
        FxForwardCurve.spot_date() bug noted above. """
    VALDATE = dt.datetime(2025, 8, 12)
    SPOT_DATE = dt.datetime(2025, 8, 14)

    def _make_curve(self, spot, for_rate, dom_rate):
        curve = FxForwardCurve(self.VALDATE, 'EUR', 'USD')
        curve.spot_date = lambda: self.SPOT_DATE
        forcurve = ConstDiscountCurve(self.VALDATE, for_rate)
        domcurve = ConstDiscountCurve(self.VALDATE, dom_rate)
        curve.load_calibrated(spot, forcurve, domcurve)
        return curve

    def test_value_at_spot_date_reproduces_input_spot(self):
        curve = self._make_curve(spot=1.10, for_rate=0.03, dom_rate=0.05)
        assert curve.value(self.SPOT_DATE) == pytest.approx(1.10)

    def test_higher_domestic_rate_gives_higher_forward(self):
        # Covered interest parity: if USD (domestic) rates > EUR (foreign) rates,
        # the EUR/USD forward should trade above spot.
        curve = self._make_curve(spot=1.10, for_rate=0.03, dom_rate=0.05)
        maturity = dt.datetime(2026, 8, 14)
        assert curve.value(maturity) > 1.10

    def test_equal_rates_gives_flat_forward(self):
        curve = self._make_curve(spot=1.10, for_rate=0.04, dom_rate=0.04)
        maturity = dt.datetime(2026, 8, 14)
        assert curve.value(maturity) == pytest.approx(1.10)


if __name__ == "__main__":
    tester = TestFxSpotDate()
    tester.test_usd_cad_one_day_lag()
