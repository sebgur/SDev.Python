import datetime as dt
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from sdevpy.utilities import timegrids
from sdevpy.utilities.schedules import SwapSchedule

DiscountFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class SwapTimes:
    """ Float64 version of swap time information necessary for pricing """
    fixed_pay: npt.NDArray[np.float64]
    fixed_alpha: npt.NDArray[np.float64]
    float_start: npt.NDArray[np.float64]
    float_end: npt.NDArray[np.float64]
    float_pay: npt.NDArray[np.float64]


def swap_times(swap: SwapSchedule, valdate: dt.datetime) -> SwapTimes:
    fx, fl = swap.fixed, swap.floating
    return SwapTimes(timegrids.model_time(valdate, fx.payment), fx.accrual_fraction,
                     timegrids.model_time(valdate, fl.adj_start), timegrids.model_time(valdate, fl.adj_end),
                     timegrids.model_time(valdate, fl.payment))


def annuity(st: SwapTimes, df_disc: DiscountFunction) -> float:
    """ PV of the fixed leg per unit of fixed rate """
    return float(np.sum(st.fixed_alpha * df_disc(st.fixed_pay)))


def float_leg_pv(st: SwapTimes, df_disc: DiscountFunction, df_proj: DiscountFunction | None = None) -> float:
    """ PV of the floating leg, forwards projected on df_proj (default: df_disc). Coupon of period j is
        alpha_j * F_j = P(start_j) / P(end_j) - 1, so the accrual fraction cancels out """
    df_proj = df_disc if df_proj is None else df_proj
    coupons = df_proj(st.float_start) / df_proj(st.float_end) - 1.0
    return float(np.sum(coupons * df_disc(st.float_pay)))


def par_rate(st: SwapTimes, df_disc: DiscountFunction, df_proj: DiscountFunction | None = None) -> float:
    """ Fixed rate that makes the swap worth zero """
    return float_leg_pv(st, df_disc, df_proj) / annuity(st, df_disc)


if __name__ == "__main__":
    from sdevpy.calibration.rates import yieldcurve as ycrv
    from sdevpy.utilities.sdaycount import DayCount
    from sdevpy.utilities.schedules import LegConvention, SwapConvention, make_spot_swap

    # Curve: linear zero rates
    valdate = dt.datetime(2026, 10, 23)
    zdates = [dt.datetime(2027, 10, 23), dt.datetime(2031, 10, 23), dt.datetime(2036, 10, 23),
              dt.datetime(2056, 10, 23)]
    zrs = np.array([0.035, 0.037, 0.040, 0.042])
    curve = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var='zerorate', interp_type='linear')
    curve.set_data(zdates, np.exp(-zrs * timegrids.model_time(valdate, zdates)))

    # USD SOFR OIS: annual ACT/360 both legs, payment 2 business days after period end
    sofr = LegConvention('1Y', DayCount.ACT360, 'USD', pay_lag=2)
    usd_ois = SwapConvention(sofr, sofr)

    # Cash flow dates of a 3Y swap
    swap = make_spot_swap(valdate, '3Y', usd_ois)
    print(f"3Y SOFR swap: start {swap.start}, maturity {swap.maturity}")
    for s, e, p, a in zip(swap.fixed.adj_start, swap.fixed.adj_end, swap.fixed.payment,
                          swap.fixed.accrual_fraction, strict=True):
        print(f"  {s} -> {e}  pay {p}  alpha {a:.6f}")

    # Par rates
    for tenor in ('1Y', '2Y', '5Y', '10Y', '30Y'):
        st = swap_times(make_spot_swap(valdate, tenor, usd_ois), valdate)
        print(f"{tenor:>4} par rate: {par_rate(st, curve.discount_float):.4%}")

    # Dual curve: EUR 6M swap, forwards projected on a curve 20bp above the discount curve
    eur_6m = SwapConvention(LegConvention('1Y', DayCount.D30_360, 'EUR'),
                            LegConvention('6M', DayCount.ACT360, 'EUR', fixing_lag=2))
    st = swap_times(make_spot_swap(valdate, '10Y', eur_6m), valdate)
    df_proj = lambda t: curve.discount_float(t) * np.exp(-0.002 * t)
    print(f"EUR 10Y par rate: single curve {par_rate(st, curve.discount_float):.4%}, "
          f"dual curve {par_rate(st, curve.discount_float, df_proj):.4%}")
