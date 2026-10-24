import datetime as dt
import numpy as np
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.utilities import schedules as sch
from sdevpy.utilities import timegrids
from sdevpy.rates import swaps

VALDATE = dt.datetime(2026, 10, 23)
SOFR = sch.LegConvention('1Y', DayCount.ACT360, 'USD', pay_lag=2)
USD_OIS = sch.SwapConvention(SOFR, SOFR)


def flat_df(rate):
    return lambda t: np.exp(-rate * np.asarray(t))


def test_swap_times():
    swap = sch.make_spot_swap(VALDATE, '2Y', USD_OIS)
    st = swaps.swap_times(swap, VALDATE)
    assert np.allclose(st.fixed_pay, timegrids.model_time(VALDATE, swap.fixed.payment), rtol=0, atol=1e-15)
    assert np.allclose(st.float_start, [4 / 365, 369 / 365], rtol=0, atol=1e-15)  # 27-Oct-26, 27-Oct-27
    assert np.allclose(st.fixed_alpha, swap.fixed.accrual_fraction, rtol=0, atol=1e-15)


def test_annuity():
    st = swaps.swap_times(sch.make_spot_swap(VALDATE, '5Y', USD_OIS), VALDATE)
    df = flat_df(0.04)
    assert abs(swaps.annuity(st, df) - np.sum(st.fixed_alpha * np.exp(-0.04 * st.fixed_pay))) < 1e-14


def test_float_leg_telescopes_without_pay_lag():
    # Single curve, payment at period end: float leg = P(start) - P(end)
    sofr0 = sch.LegConvention('1Y', DayCount.ACT360, 'USD')
    st = swaps.swap_times(sch.make_spot_swap(VALDATE, '10Y', sch.SwapConvention(sofr0, sofr0)), VALDATE)
    df = flat_df(0.04)
    ref = df(st.float_start[0]) - df(st.float_end[-1])
    assert abs(swaps.float_leg_pv(st, df) - ref) < 1e-14


def test_one_period_par_rate():
    # One annual period, same payment date on both legs: K = (P(s)/P(e) - 1) / alpha
    st = swaps.swap_times(sch.make_spot_swap(VALDATE, '1Y', USD_OIS), VALDATE)
    df = flat_df(0.035)
    ref = (df(st.float_start[0]) / df(st.float_end[0]) - 1.0) / st.fixed_alpha[0]
    assert abs(swaps.par_rate(st, df) - ref) < 1e-14


def test_dual_curve():
    eur = sch.SwapConvention(sch.LegConvention('1Y', DayCount.D30_360, 'EUR'),
                             sch.LegConvention('6M', DayCount.ACT360, 'EUR', fixing_lag=2))
    st = swaps.swap_times(sch.make_spot_swap(VALDATE, '10Y', eur), VALDATE)
    disc = flat_df(0.04)
    # Same projection and discount curves: identical to single curve
    assert abs(swaps.par_rate(st, disc, disc) - swaps.par_rate(st, disc)) < 1e-15
    # Projection curve 20bp higher (continuous): par rate about 20bp higher
    spread = swaps.par_rate(st, disc, flat_df(0.042)) - swaps.par_rate(st, disc)
    assert 0.0019 < spread < 0.0022


def test_par_rates_regression():
    from sdevpy.calibration.rates import yieldcurve as ycrv
    zdates = [dt.datetime(2027, 10, 23), dt.datetime(2031, 10, 23), dt.datetime(2036, 10, 23),
              dt.datetime(2056, 10, 23)]
    zrs = np.array([0.035, 0.037, 0.040, 0.042])
    curve = ycrv.InterpolatedYieldCurve(valdate=VALDATE, interp_var='zerorate', interp_type='linear')
    curve.set_data(zdates, np.exp(-zrs * timegrids.model_time(VALDATE, zdates)))

    test = [swaps.par_rate(swaps.swap_times(sch.make_spot_swap(VALDATE, t, USD_OIS), VALDATE), curve.discount_float)
            for t in ('1Y', '2Y', '5Y', '10Y', '30Y')]
    ref = [0.035137421956, 0.035643766953, 0.037112473365, 0.039864628649, 0.041521246891]
    assert np.allclose(test, ref, rtol=0, atol=1e-11)
