import datetime as dt
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from sdevpy.utilities import timegrids
from sdevpy.utilities.schedules import BondScheduler, to_date, BondConvention, coupons_per_year, make_bond_scheduler


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
    simple_final_period: bool = False  # simple-interest yield when only the last coupon is left

    @property
    def simple_flows(self) -> npt.NDArray[np.bool_]:
        """ Flows discounted at simple interest in the yield formulas: the last one, in the final period """
        return np.full(len(self.periods), self.simple_final_period and len(self.periods) == 1)


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
    return BondFlows(settle, payment, alpha, periods, accrued, bond.coupons_per_year, pay_times, settle_time,
                     bond.conv.simple_final_period)


def cashflows(bf: BondFlows, coupon: float, redemption: float = 100.0) -> npt.NDArray[np.float64]:
    """ Cash flow amounts per 100 notional """
    amounts = 100.0 * coupon * bf.alpha
    amounts[-1] += redemption
    return amounts


def accrued_interest(bf: BondFlows, coupon: float) -> float:
    return 100.0 * coupon * bf.accrued_fraction


def _yield_discount(tau, f, y, simple):
    """ Street-convention discount factor of each flow at yield y and its first two derivatives in y:
        (1 + y/f)^-tau, or 1 / (1 + y tau/f) where simple. Scalars or arrays (one entry per flow) """
    base = 1.0 + y / f
    v = base ** -tau
    dv = -tau / f * v / base
    d2v = tau * (tau + 1.0) / (f * f) * v / (base * base)
    if np.any(simple):
        a = tau / f
        vs = 1.0 / (1.0 + a * y)
        v = np.where(simple, vs, v)
        dv = np.where(simple, -a * vs ** 2, dv)
        d2v = np.where(simple, 2.0 * a * a * vs ** 3, d2v)
    return v, dv, d2v


def dirty_from_yield(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ Street convention: each cash flow discounted at (1 + y/f) per coupon period, at simple interest
        in the final coupon period when the bond's convention says so """
    v = _yield_discount(bf.periods, bf.frequency, y, bf.simple_flows)[0]
    return float(np.sum(cashflows(bf, coupon, redemption) * v))


def yield_from_dirty(bf: BondFlows, coupon: float, dirty: float, redemption: float = 100.0,
                     guess: float = 0.05, tol: float = 1e-12, max_iter: int = 50) -> float:
    """ Yield to maturity from dirty price (Newton) """
    amounts, simple = cashflows(bf, coupon, redemption), bf.simple_flows
    y = guess
    for _ in range(max_iter):
        v, dv, _ = _yield_discount(bf.periods, bf.frequency, y, simple)
        err = np.sum(amounts * v) - dirty
        deriv = np.sum(amounts * dv)
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
    pv = cashflows(bf, coupon, redemption) * _yield_discount(bf.periods, bf.frequency, y, bf.simple_flows)[0]
    return float(np.sum(pv * bf.periods) / np.sum(pv) / bf.frequency)


def modified_duration(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ -(1/P) dP/dy: relative price change per unit change in yield """
    amounts = cashflows(bf, coupon, redemption)
    v, dv, _ = _yield_discount(bf.periods, bf.frequency, y, bf.simple_flows)
    return float(-np.sum(amounts * dv) / np.sum(amounts * v))


def convexity(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ (1/P) d2P/dy2 """
    amounts = cashflows(bf, coupon, redemption)
    v, _, d2v = _yield_discount(bf.periods, bf.frequency, y, bf.simple_flows)
    return float(np.sum(amounts * d2v) / np.sum(amounts * v))


def dv01(bf: BondFlows, coupon: float, y: float, redemption: float = 100.0) -> float:
    """ Dirty price gain (per 100 notional) for a 1bp fall in yield """
    return modified_duration(bf, coupon, y, redemption) * dirty_from_yield(bf, coupon, y, redemption) * 1e-4


def dirty_from_zspread(bf: BondFlows, coupon: float, df: DiscountFunction, z: float,
                       redemption: float = 100.0) -> float:
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


def convert_yield(y: float, from_frequency: int, to_frequency: int) -> float:
    """ Same yield under another compounding frequency: (1 + y1/f1)^f1 = (1 + y2/f2)^f2 """
    if from_frequency == to_frequency:
        return y
    return to_frequency * ((1.0 + y / from_frequency) ** (from_frequency / to_frequency) - 1.0)


def par_yield(maturity: dt.date, settlement: dt.date, conv: BondConvention, df: DiscountFunction,
              valdate: dt.datetime) -> float:
    """ Yield of a hypothetical bond with conventions conv, issued at settlement and maturing on maturity
        (short first coupon), whose coupon prices it at par on the curve """
    bond = make_bond_scheduler(to_date(settlement), to_date(maturity), conv)
    bf = bond_flows(bond, settlement, valdate)
    c = par_coupon(bf, df)
    return yield_from_dirty(bf, c, 100.0, guess=c)


def g_spread(bf: BondFlows, coupon: float, dirty: float, df: DiscountFunction, maturity: dt.date,
             gov_conv: BondConvention, valdate: dt.datetime, redemption: float = 100.0) -> float:
    """ Bond yield minus the government par yield at the bond's maturity, both in the government's compounding
        (e.g. semi-annual for UST). df: the government curve """
    gov_par = par_yield(maturity, bf.settlement, gov_conv, df, valdate)
    y = yield_from_dirty(bf, coupon, dirty, redemption)
    return convert_yield(y, bf.frequency, coupons_per_year(gov_conv.leg.frequency)) - gov_par


###################### FOR CURVE CONSTRUCTION #########################################################################
@dataclass(frozen=True)
class FlatFlows:
    """ Cash flows of n bonds concatenated, bond i owning flows starts[i]:starts[i+1]. The curve is needed only
        at the distinct pay/settlement times: flow j pays at times[flow_index[j]], bond i settles at
        times[settle_index[i]] """
    times: npt.NDArray[np.float64]  # (U,) distinct pay and settlement times, increasing
    amounts: npt.NDArray[np.float64]  # (N,) per 100 notional, coupon and redemption included
    starts: npt.NDArray[np.intp]  # (n,) index of each bond's first flow
    owner: npt.NDArray[np.intp]  # (N,) bond index of each flow
    flow_index: npt.NDArray[np.intp]  # (N,) index in times of each flow's pay time
    settle_index: npt.NDArray[np.intp]  # (n,) index in times of each bond's settlement time
    periods: npt.NDArray[np.float64]  # (N,) coupon periods from settlement (street-yield exponents)
    frequency: npt.NDArray[np.float64]  # (n,) coupons per year
    simple: npt.NDArray[np.bool_]  # (N,) flows discounted at simple interest in the yield formulas

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
    all_times = np.concatenate([bf.pay_times for bf in flows] + [np.array([bf.settle_time for bf in flows])])
    times, index = np.unique(all_times, return_inverse=True)
    n = len(amounts)
    periods = np.concatenate([bf.periods for bf in flows])
    frequency = np.array([bf.frequency for bf in flows], dtype=float)
    simple = np.concatenate([bf.simple_flows for bf in flows])
    return FlatFlows(times, amounts, starts, owner, index[:n], index[n:], periods, frequency, simple)


def dirty_from_yields(ff: FlatFlows, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Street-convention dirty prices of all bonds from their yields """
    v = _yield_discount(ff.periods, ff.frequency[ff.owner], y[ff.owner], ff.simple)[0]
    return np.add.reduceat(ff.amounts * v, ff.starts)


def yields_from_dirty(ff: FlatFlows, dirty: npt.NDArray[np.float64], guess: float = 0.05, tol: float = 1e-12,
                      max_iter: int = 50) -> npt.NDArray[np.float64]:
    """ Street-convention yields of all bonds from their dirty prices (Newton, all bonds at once) """
    f = ff.frequency[ff.owner]
    y = np.full(ff.n_bonds, guess)
    for _ in range(max_iter):
        v, dv, _ = _yield_discount(ff.periods, f, y[ff.owner], ff.simple)
        err = np.add.reduceat(ff.amounts * v, ff.starts) - dirty
        deriv = np.add.reduceat(ff.amounts * dv, ff.starts)
        step = err / deriv
        y -= step
        if np.max(np.abs(step)) < tol:
            return y
    raise RuntimeError("Yields did not converge")


def dv01s(ff: FlatFlows, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Dirty price gain (per 100 notional) for a 1bp fall in yield, all bonds """
    dv = _yield_discount(ff.periods, ff.frequency[ff.owner], y[ff.owner], ff.simple)[1]
    return -1e-4 * np.add.reduceat(ff.amounts * dv, ff.starts)


def dirty_prices(ff: FlatFlows, df: DiscountFunction) -> npt.NDArray[np.float64]:
    """ Dirty prices at each bond's settlement date """
    return prices_from_discounts(ff, df(ff.times))


def prices_from_discounts(ff: FlatFlows, d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """ Dirty prices given the discount factors at ff.times """
    return np.add.reduceat(ff.amounts * d[ff.flow_index], ff.starts) / d[ff.settle_index]


def prices_jac(ff: FlatFlows, d: npt.NDArray[np.float64], g: npt.NDArray[np.float64],
               dv: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """ Dirty prices (n_bonds,) and their Jacobian (n_bonds, n_params), given the discount factors d at
        ff.times and their derivatives in factored form dDF/dparams = g[:, None] * dv.
        P = S / D_settle, S = sum a DF; dS = A dv with A[i, u] = sum of a g over bond i's flows paid at times[u]
        (sparse; repeated (i, u) entries are summed) """
    p = prices_from_discounts(ff, d)
    agg = sps.csr_matrix((ff.amounts * g[ff.flow_index], ff.flow_index, np.append(ff.starts, ff.n_flows)),
                         shape=(ff.n_bonds, len(ff.times)))
    ds = agg @ dv
    s = ff.settle_index
    return p, (ds - (p * g[s])[:, None] * dv[s]) / d[s, None]


if __name__ == "__main__":
    from sdevpy.calibration.rates import yieldcurve as ycrv
    # from sdevpy.utilities.scalendar import BDC
    # from sdevpy.utilities.sdaycount import DayCount
    from sdevpy.utilities.schedules import make_bond_scheduler
    from sdevpy.conventions.bondconventions import UST

    # US Treasury-style bond: 4% semi-annual, ACT/ACT ICMA, unadjusted accrual, end-of-month rule
    ust = UST # LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
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
