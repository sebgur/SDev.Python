import datetime as dt
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from sdevpy.utilities import timegrids
from sdevpy.utilities.schedules import BondScheduler, to_date

DiscountFunction = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


@dataclass(frozen=True)
class BondFlows:
    """ Coupon-independent description of what a buyer settling on `settlement` receives """
    settlement: dt.date
    payment: npt.NDArray[np.datetime64]  # payment dates of the remaining coupons (redemption with the last)
    alpha: npt.NDArray[np.float64]  # accrual fraction of each remaining coupon
    periods: npt.NDArray[np.float64]  # coupon periods from settlement to each coupon date (yield exponents)
    accrued_fraction: float  # accrual fraction from the last coupon date to settlement
    frequency: int  # coupons per year
    pay_times: npt.NDArray[np.float64] | None = None  # model times from valdate (curve pricing only)
    settle_time: float | None = None


def bond_flows(bond: BondScheduler, settlement: dt.date, valdate: dt.datetime | None = None) -> BondFlows:
    """ Remaining cash flows at settlement. valdate (the curve's valuation date) is only needed for curve pricing """
    settle = to_date(settlement)
    starts, ends = bond.accrual_dates()
    i = int(np.searchsorted(ends, np.datetime64(settle, 'D'), side='right'))  # first coupon not yet paid
    if i == len(ends):
        raise ValueError(f"Bond matured before settlement {settle}")

    last_coupon = to_date(starts[i])
    accrued = bond.accrual_fraction(last_coupon, settle) if settle > last_coupon else 0.0
    alpha = bond.schedule.accrual_fraction[i:]
    remaining = np.concatenate(([alpha[0] - accrued], alpha[1:]))  # accrual left in each period
    periods = bond.coupons_per_year * np.cumsum(remaining)

    payment = bond.schedule.payment[i:]
    pay_times = settle_time = None
    if valdate is not None:
        pay_times = timegrids.model_time(valdate, payment)
        settle_time = timegrids.model_time(valdate, np.datetime64(settle, 'D'))
    return BondFlows(settle, payment, alpha, periods, accrued, bond.coupons_per_year, pay_times, settle_time)


def cashflows(bf: BondFlows, coupon: float, redemption: float = 100.0) -> npt.NDArray[np.float64]:
    """ Cash flow amounts per 100 notional """
    amounts = 100.0 * coupon * bf.alpha
    amounts[-1] += redemption
    return amounts


def accrued_interest(bf: BondFlows, coupon: float) -> float:
    return 100.0 * coupon * bf.accrued_fraction


def dirty_from_yield(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ Street convention: each cash flow discounted at (1 + y/f) per coupon period """
    return float(np.sum(cashflows(bf, coupon, redemption) * (1.0 + y / bf.frequency) ** -bf.periods))


def yield_from_dirty(bf: BondFlows, coupon: float, dirty: float, redemption: float = 100.0,
                     guess: float = 0.05, tol: float = 1e-12, max_iter: int = 50) -> float:
    """ Yield to maturity from dirty price (Newton) """
    amounts, f, tau = cashflows(bf, coupon, redemption), bf.frequency, bf.periods
    y = guess
    for _ in range(max_iter):
        v = (1.0 + y / f) ** -tau
        err = np.sum(amounts * v) - dirty
        deriv = -np.sum(amounts * tau / f * v / (1.0 + y / f))
        step = err / deriv
        y -= step
        if abs(step) < tol:
            return float(y)
    raise RuntimeError("Yield did not converge")


def dirty_from_curve(bf: BondFlows, coupon: float, df: DiscountFunction, redemption: float = 100.0) -> float:
    """ Dirty price at settlement: PV of the cash flows, forward to the settlement date """
    if bf.pay_times is None:
        raise ValueError("BondFlows built without valdate: no times for curve pricing")
    return float(np.sum(cashflows(bf, coupon, redemption) * df(bf.pay_times)) / df(np.array([bf.settle_time]))[0])


def par_coupon(bf: BondFlows, df: DiscountFunction, redemption: float = 100.0) -> float:
    """ Coupon rate that prices the bond at par (clean price 100) on the curve """
    if bf.pay_times is None:
        raise ValueError("BondFlows built without valdate: no times for curve pricing")
    disc = df(bf.pay_times) / df(np.array([bf.settle_time]))[0]
    return float((100.0 - redemption * disc[-1]) / (100.0 * (np.sum(bf.alpha * disc) - bf.accrued_fraction)))


def macaulay_duration(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ PV-weighted average time to the cash flows, in years (street-convention yield discounting) """
    pv = cashflows(bf, coupon, redemption) * (1.0 + y / bf.frequency) ** -bf.periods
    return float(np.sum(pv * bf.periods) / np.sum(pv) / bf.frequency)


def modified_duration(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ -(1/P) dP/dy: relative price change per unit change in yield """
    return macaulay_duration(bf, coupon, y, redemption) / (1.0 + y / bf.frequency)


def convexity(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ (1/P) d2P/dy2 """
    f = bf.frequency
    pv = cashflows(bf, coupon, redemption) * (1.0 + y / f) ** -bf.periods
    return float(np.sum(pv * bf.periods * (bf.periods + 1.0)) / (f * f * (1.0 + y / f) ** 2) / np.sum(pv))


def dv01(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ Dirty price gain (per 100 notional) for a 1bp fall in yield """
    return modified_duration(bf, coupon, y, redemption) * dirty_from_yield(bf, coupon, y, redemption) * 1e-4


def dirty_from_zspread(bf: BondFlows, coupon: float, df: DiscountFunction, z: float, redemption: float = 100.0) -> float:
    """ Curve price with a constant continuously-compounded spread z over the curve's zero rates """
    if bf.pay_times is None:
        raise ValueError("BondFlows built without valdate: no times for curve pricing")
    disc = df(bf.pay_times) / df(np.array([bf.settle_time]))[0] * np.exp(-z * (bf.pay_times - bf.settle_time))
    return float(np.sum(cashflows(bf, coupon, redemption) * disc))


def z_spread(bf: BondFlows, coupon: float, dirty: float, df: DiscountFunction, redemption: float = 100.0,
             tol: float = 1e-12, max_iter: int = 50) -> float:
    """ Constant spread over the curve that reprices the bond to its market dirty price (Newton) """
    if bf.pay_times is None:
        raise ValueError("BondFlows built without valdate: no times for curve pricing")
    amounts = cashflows(bf, coupon, redemption)
    tau = bf.pay_times - bf.settle_time
    base = amounts * df(bf.pay_times) / df(np.array([bf.settle_time]))[0]
    z = 0.0
    for _ in range(max_iter):
        pv = base * np.exp(-z * tau)
        step = (np.sum(pv) - dirty) / -np.sum(pv * tau)
        z -= step
        if abs(step) < tol:
            return float(z)
    raise RuntimeError("Z-spread did not converge")


###################### FOR CURVE CONSTRUCTION #########################################################################
@dataclass(frozen=True)
class FlatFlows:
    """ Cash flows of n bonds concatenated, bond i owning flows starts[i]:starts[i+1].
        times = pay times of all flows (N), then settlement times of all bonds (n): one curve call per pricing """
    times: npt.NDArray[np.float64]  # (N + n,)
    amounts: npt.NDArray[np.float64]  # (N,) per 100 notional, coupon and redemption included
    starts: npt.NDArray[np.intp]  # (n,) index of each bond's first flow
    owner: npt.NDArray[np.intp]  # (N,) bond index of each flow

    @property
    def n_bonds(self) -> int:
        return len(self.starts)

    @property
    def n_flows(self) -> int:
        return len(self.amounts)


def flatten(flows: list[BondFlows], coupons: npt.ArrayLike, redemptions: npt.ArrayLike = 100.0) -> FlatFlows:
    """ Concatenate the flows of several bonds/bills (all built with the same valdate) """
    if any(bf.pay_times is None for bf in flows):
        raise ValueError("BondFlows built without valdate: no times for curve pricing")
    coupons = np.broadcast_to(np.asarray(coupons, dtype=float), (len(flows),))
    redemptions = np.broadcast_to(np.asarray(redemptions, dtype=float), (len(flows),))

    sizes = np.array([len(bf.alpha) for bf in flows])
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    owner = np.repeat(np.arange(len(flows)), sizes)
    amounts = 100.0 * np.repeat(coupons, sizes) * np.concatenate([bf.alpha for bf in flows])
    amounts[starts + sizes - 1] += redemptions
    times = np.concatenate([bf.pay_times for bf in flows] + [np.array([bf.settle_time for bf in flows])])
    return FlatFlows(times, amounts, starts, owner)


def dirty_prices(ff: FlatFlows, df: DiscountFunction) -> npt.NDArray[np.float64]:
    """ Dirty prices at each bond's settlement date """
    return prices_from_discounts(ff, df(ff.times))
    # d = df(ff.times)
    # n_flows = ff.n_flows
    # return np.add.reduceat(ff.amounts * d[:n_flows], ff.starts) / d[n_flows:]


def prices_from_discounts(ff: FlatFlows, d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Dirty prices given the discount factors at ff.times (dirty_prices(ff, df) = this with d = df(ff.times)) """
    n = ff.n_flows
    return np.add.reduceat(ff.amounts * d[:n], ff.starts) / d[n:]


def prices_jac(ff: FlatFlows, d: npt.NDArray[np.float64], g: npt.NDArray[np.float64],
               dv: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ Dirty prices (n_bonds,) and their Jacobian (n_bonds, n_params), given the discount factors d at
        ff.times and their derivatives in factored form dDF/dparams = g[:, None] * dv.
        P = S / D_settle, S = sum a DF; the flow sums dS = A diag(a g) dv go through a sparse matrix """
    n = ff.n_flows
    p = prices_from_discounts(ff, d)
    agg = sps.csr_matrix((ff.amounts * g[:n], np.arange(n), np.append(ff.starts, n)), shape=(ff.n_bonds, n))
    ds = agg @ dv[:n]
    dd_settle = g[n:, None] * dv[n:]
    return p, (ds - p[:, None] * dd_settle) / d[n:, None]


if __name__ == "__main__":
    from sdevpy.calibration.rates import yieldcurve as ycrv
    from sdevpy.utilities.scalendar import BDC
    from sdevpy.utilities.sdaycount import DayCount
    from sdevpy.utilities.schedules import LegConvention, make_bond_scheduler

    # US Treasury-style bond: 4% semi-annual, ACT/ACT ICMA, unadjusted accrual, end-of-month rule
    ust = LegConvention('6M', DayCount.ACTACT_ICMA, 'USD', bdc=BDC.F, adjust_accrual=False, eom=True)
    bond = make_bond_scheduler(dt.date(2026, 2, 15), dt.date(2036, 2, 15), ust)
    coupon = 0.04

    # Remaining cash flows when settling on 15-May-2026
    valdate = dt.datetime(2026, 5, 14)
    settlement = dt.date(2026, 5, 15)
    bf = bond_flows(bond, settlement, valdate)
    print(f"Settlement {bf.settlement}: {len(bf.payment)} remaining coupons, "
          f"accrued interest {accrued_interest(bf, coupon):.6f}")
    for p, a, n in list(zip(bf.payment, cashflows(bf, coupon), bf.periods, strict=True))[:3]:
        print(f"  pay {p}  amount {a:.6f}  periods from settlement {n:.6f}")

    # Price <-> yield
    y = 0.045
    dirty = dirty_from_yield(bf, coupon, y)
    clean = dirty - accrued_interest(bf, coupon)
    print(f"Yield {y:.4%}: dirty {dirty:.6f}, clean {clean:.6f}")
    print(f"Yield back from dirty price: {yield_from_dirty(bf, coupon, dirty):.6%}")

    # Curve: linear zero rates
    zdates = [dt.datetime(2027, 5, 14), dt.datetime(2031, 5, 14), dt.datetime(2036, 5, 14)]
    zrs = np.array([0.035, 0.037, 0.040])
    curve = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var='zerorate', interp_type='linear')
    curve.set_data(zdates, np.exp(-zrs * timegrids.model_time(valdate, zdates)))

    # Price on the curve, and the coupon that would price the bond at par
    dirty = dirty_from_curve(bf, coupon, curve.discount_float)
    print(f"Curve price: dirty {dirty:.6f}, clean {dirty - accrued_interest(bf, coupon):.6f}, "
          f"yield {yield_from_dirty(bf, coupon, dirty):.4%}")
    par = par_coupon(bf, curve.discount_float)
    print(f"Par coupon: {par:.4%}")
