import pytest
import datetime as dt
import numpy as np
from sdevpy.pricingcontext import default_market_repository, default_calibration_repository
from sdevpy.market.source import MarketDataSource
from sdevpy.market.dataset import MarketDataSet
from sdevpy.calibration.source import CalibrationDataSource
from sdevpy.calibration.dataset import CalibrationDataSet
from sdevpy.calibration.eq import eqvolsurface as eqvs
from sdevpy.calibration.eq import eqforward as eqf
from sdevpy.market.fixings import FixingHandler, data_file
from sdevpy.market.spot import SpotData


class ConstDiscountCurve:
    """ Test double: flat continuously-compounded discount curve, tagged with the name
        it was requested under so tests can check routing as well as the math. """
    def __init__(self, name, valdate, rate=0.0):
        self.name = name
        self.valdate = valdate
        self.rate = rate

    def discount(self, date):
        t = (date - self.valdate).days / 365.0
        return np.exp(-self.rate * t)

    def discount_float(self, t):
        return np.exp(-self.rate * t)


class FakeSource(MarketDataSource):
    def __init__(self, spots):
        self._spots = spots

    def get_spot_data(self, name, date):
        return SpotData(date, self._spots[name])

    def _not_used(self, *args, **kwargs):
        raise NotImplementedError("FakeSource only supports get_spot_data for these tests")

    get_fixing_handler = _not_used
    get_correlations = _not_used
    get_eq_forward_data = _not_used
    get_eq_vol_data = _not_used
    get_fx_vol_data = _not_used


class FakeCalibSource(CalibrationDataSource):
    def __init__(self, valdate=None, rates=None):
        self._valdate = valdate
        self._rates = rates or {}

    def get_yieldcurve(self, name, date):
        return ConstDiscountCurve(name, self._valdate or date, self._rates.get(name, 0.0))

    def _not_used(self, *args, **kwargs):
        raise NotImplementedError("FakeCalibSource only supports get_yieldcurve for these tests")

    get_impliedvol_data = _not_used
    get_localvol_data = _not_used
    get_fxvol_data = _not_used
    save_impliedvol_data = _not_used
    save_localvol_data = _not_used
    save_fxvol_data = _not_used


###################################################################################################

def _make_handler(interpolate=False):
    dates = [dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 2), dt.datetime(2025, 12, 3)]
    values = [100.0, 101.0, 102.0]
    return FixingHandler("TEST", dates, values, interpolate=interpolate)


def test_fixinghandler_scalar_lookup():
    h = _make_handler()
    assert h.value(dt.datetime(2025, 12, 2)) == 101.0


def test_fixinghandler_list_lookup():
    h = _make_handler()
    result = h.value([dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 3)])
    assert result == [100.0, 102.0]


def test_fixinghandler_missing_raises():
    h = _make_handler(interpolate=False)
    with pytest.raises(ValueError):
        h.value(dt.datetime(2025, 12, 5))


def test_fixinghandler_unsorted_input_stores_sorted():
    dates = [dt.datetime(2025, 12, 3), dt.datetime(2025, 12, 1), dt.datetime(2025, 12, 2)]
    values = [102.0, 100.0, 101.0]
    h = FixingHandler("TEST", dates, values)
    assert h.dates[0] == dt.datetime(2025, 12, 1)
    assert h.values[0] == 100.0


def test_fixinghandler_interpolation():
    h = _make_handler(interpolate=True)
    # dt.datetime(2025, 12, 1) and (2025, 12, 3) are exact; interpolated mid should be ~101
    result = h.value(dt.datetime(2025, 12, 2))
    assert abs(result - 101.0) < 0.5


def test_data_file_returns_correct_path():
    from pathlib import Path
    p = data_file("ABC", folder=Path("/some/folder"))
    assert p == Path("/some/folder/ABC.csv")


def test_correlations():
    names = ['ABC', 'KLM', 'XYZ']
    valdate = dt.datetime(2025, 12, 15)
    mkt = default_market_repository()[valdate]
    c = mkt.get_correlations(names)
    ref = np.asarray([0.5, 0.1, 0.1])
    test = np.asarray([c[0, 1], c[0, 2], c[1, 2]])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_spotdata():
    name, valdate = "ABC", dt.datetime(2025, 12, 15)

    # Fetch data
    mkt = default_market_repository()[valdate]
    test = mkt.get_spot(name)
    ref = 100.0
    assert test == ref


def test_eqforward_creation():
    name, valdate = "ABC", dt.datetime(2025, 12, 15)
    spot = 100.0

    # Get data from existing file
    mkt = default_market_repository()[valdate]
    test_data = mkt.get_eq_forward_data(name)

    # Create forward curve
    curve = eqf.EqForwardCurve(valdate=valdate, interp_var='forward', interp_type='cubicspline')
    yieldcurve = default_calibration_repository()[valdate].get_yieldcurve('USD.SOFR.1D')
    curve.calibrate(test_data, spot, yieldcurve)

    # Interpolate and display
    test_dates = [dt.datetime(2026, 3, 15), dt.datetime(2026, 8, 15), dt.datetime(2027, 2, 15),
                  dt.datetime(2031, 2, 15), dt.datetime(2036, 2, 15)]

    test = curve.value(test_dates)
    ref = np.asarray([100.50233117, 101.4947209, 102.06711626, 112.73343447, 128.4201])
    # ref = np.asarray([100.28716535, 101.09321315, 102.21072051, 111.87417472, 128.4201])
    assert np.allclose(test, ref, rtol=0.0, atol=1e-8)


def test_eq_option_strikes():
    name, valdate = "ABC", dt.datetime(2025, 12, 15)

    # Retrieve market option data object
    mkt = default_market_repository()[valdate]
    vol_data = mkt.get_eq_vol_data(name)

    # Retrieve forward curve
    fwd_curve = mkt.get_eq_forward_curves([name])[0]

    # Access data in object
    test = eqvs.get_strikes(vol_data, fwd_curve, 'absolute')
    # print(test)

    ref = np.asarray([[90.26318122, 94.70076604, 99.88756326, 105.35844335, 110.53815253],
                      [84.95857122, 91.65636925, 99.71914514, 108.49118276, 117.04419888],
                      [79.28085982, 88.26212825, 99.43907907, 112.03140738, 124.72279524],
                      [71.77471782, 83.53767673, 98.88130446, 117.04314454, 136.22502001],
                      [62.15139633, 77.03054626, 97.77512372, 124.10628358, 153.81753883],
                      [46.17835038, 64.83651535, 94.53027807, 137.82316066, 193.51001927]])
    assert(np.allclose(test, ref, rtol=0.0, atol=1e-8))

    test = eqvs.get_strikes(vol_data, fwd_curve, 'relative')
    # print(test)

    ref = np.asarray([[0.90082835, 0.94511554, 0.99687988, 1.05147937, 1.10317297],
                      [0.84534839, 0.91199231, 0.99221794, 1.07950081, 1.16460439],
                      [0.78492002, 0.87383905, 0.98449644, 1.10916676, 1.23481783],
                      [0.70353483, 0.81883520, 0.96923323, 1.14725535, 1.33527584],
                      [0.59714405, 0.74010135, 0.93941306, 1.19240007, 1.47786267],
                      [0.41783899, 0.58666505, 0.85534533, 1.24707553, 1.75095106]])
    assert(np.allclose(test, ref, rtol=0.0, atol=1e-8))


class TestGetFxSpot:
    VALDATE = dt.datetime(2025, 12, 15)

    def _mkt(self, spots):
        return MarketDataSet(self.VALDATE, FakeSource(spots))

    def test_direct_usd_pair_conventional_order(self):
        mkt = self._mkt({'EURUSD': 1.10})
        assert mkt.get_fx_spot('EUR', 'USD') == pytest.approx(1.10)

    def test_direct_usd_pair_reversed_request_inverts(self):
        mkt = self._mkt({'EURUSD': 1.10})
        result = mkt.get_fx_spot('USD', 'EUR')
        assert result == pytest.approx(1.0 / 1.10)

    def test_base_side_usd_pair(self):
        mkt = self._mkt({'USDJPY': 150.0})
        # provider = FakeProvider({'USDJPY': 150.0})
        assert mkt.get_fx_spot('USD', 'JPY') == pytest.approx(150.0)

    def test_cross_triangulates_through_usd(self):
        mkt = self._mkt({'EURUSD': 1.10, 'USDJPY': 150.0})
        result = mkt.get_fx_spot('EUR', 'JPY')
        assert result == pytest.approx(1.10 * 150.0)  # EURJPY = EURUSD * USDJPY

    def test_same_currency_returns_one(self):
        mkt = self._mkt({})
        assert mkt.get_fx_spot('EUR', 'EUR') == 1.0


class TestXccyCurve:
    VALDATE = dt.datetime(2025, 12, 15)

    def _calib(self):
        return CalibrationDataSet(self.VALDATE, FakeCalibSource())

    def test_usd_returns_rfr_curve(self):
        assert self._calib().get_xccycurve('USD').name == 'USD.SOFR.1D'

    def test_non_usd_returns_xccy_curve(self):
        assert self._calib().get_xccycurve('EUR').name == 'EUR.XCCY'

    def test_unknown_rfr_currency_raises(self):
        with pytest.raises(ValueError):
            self._calib().get_rfrcurve('JPY')


def test_get_fx_forward_curve():
    valdate = dt.datetime(2025, 8, 12)
    mkt = MarketDataSet(valdate, FakeSource({'EURUSD': 1.10}))
    calib = CalibrationDataSet(valdate, FakeCalibSource(valdate=valdate, rates={'EUR.XCCY': 0.03, 'USD.SOFR.1D': 0.05}))
    curve = calib.get_fx_forward_curve('EURUSD', mkt)
    assert curve.value(curve.spot_date()) == pytest.approx(1.10)


def test_get_fx_forward_curve_rejects_mismatched_market_date():
    calib = CalibrationDataSet(dt.datetime(2025, 8, 12), FakeCalibSource())
    mkt = MarketDataSet(dt.datetime(2025, 8, 13), FakeSource({'EURUSD': 1.10}))
    with pytest.raises(ValueError):
        calib.get_fx_forward_curve('EURUSD', mkt)


if __name__ == "__main__":
    test_eq_option_strikes()
