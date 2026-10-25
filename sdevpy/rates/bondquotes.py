import datetime as dt
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sdevpy.utilities import timegrids
from sdevpy.utilities.scalendar import make_calendar
from sdevpy.utilities.schedules import BondConvention, to_date
from sdevpy.rates.bonds import BondFlows, accrued_interest, dirty_from_yield, yield_from_dirty, dv01


BILL_FREQUENCY = 2  # yield exponents of bills measured in half-years

class QuoteType(Enum):
    CLEAN = "clean"
    DIRTY = "dirty"
    YIELD = "yield"  # street-convention yield to maturity
    DISCOUNT = "discount"  # bill discount rate


def settlement_date(trade_date: dt.date, conv: BondConvention) -> dt.date:
    """ Settlement lag business days after trade date (UST: T+1) """
    return make_calendar(conv.leg.calendar).add_business_days(to_date(trade_date), conv.settlement_lag)


def bill_flows(maturity: dt.date, settlement: dt.date, valdate: dt.datetime | None = None) -> BondFlows:
    """ Zero-coupon bill as a single flow of 100 at maturity (alpha = 0: no coupon) """
    settle, pay = to_date(settlement), np.array([np.datetime64(to_date(maturity), 'D')])
    days = float((pay[0] - np.datetime64(settle, 'D')).astype(int))
    if days <= 0:
        raise ValueError(f"Bill matured before settlement {settle}")

    periods = np.array([BILL_FREQUENCY * days / 365.0])
    pay_times = settle_time = None
    if valdate is not None:
        pay_times = timegrids.model_time(valdate, pay)
        settle_time = timegrids.model_time(valdate, np.datetime64(settle, 'D'))
    return BondFlows(settle, pay, np.zeros(1), periods, 0.0, BILL_FREQUENCY, pay_times, settle_time, days <= 182)


def bill_days(bf: BondFlows) -> int:
    return int((bf.payment[-1] - np.datetime64(bf.settlement, 'D')).astype(int))


def price_from_discount_rate(bf: BondFlows, d: float) -> float:
    """ Bill price per 100 from its discount rate (ACT/360) """
    return 100.0 * (1.0 - d * bill_days(bf) / 360.0)


def discount_rate_from_price(bf: BondFlows, price: float) -> float:
    return (1.0 - price / 100.0) * 360.0 / bill_days(bf)


################## QUOTES #########################################################################
@dataclass(frozen=True)
class BondQuote:
    """ A market quote on a bond or bill. coupon = 0 for bills """
    flows: BondFlows
    coupon: float
    quote_type: QuoteType
    value: float
    redemption: float = 100.0


def target_dirty(q: BondQuote) -> float:
    """ Dirty price per 100 at the quote's settlement date """
    bf = q.flows
    match q.quote_type:
        case QuoteType.DIRTY:
            return q.value
        case QuoteType.CLEAN:
            return q.value + accrued_interest(bf, q.coupon)
        case QuoteType.YIELD:
            return dirty_from_yield(bf, q.coupon, q.value, q.redemption)
        case QuoteType.DISCOUNT:
            return price_from_discount_rate(bf, q.value)
        case _:
            raise ValueError(f"Unsupported quote type: {q.quote_type}")


def fit_weight(q: BondQuote, dirty: float) -> float:
    """ 1/DV01: turns a price residual into an (approximate) yield residual in bp """
    y = q.value if q.quote_type == QuoteType.YIELD else yield_from_dirty(q.flows, q.coupon, dirty, q.redemption)
    return 1.0 / dv01(q.flows, q.coupon, y, q.redemption)


def parse_32nds(quote: str) -> float:
    """ Treasury price in 32nds: '99-16' = 99 + 16/32, '99-16+' = 99 + 16.5/32, '99-162' = 99 + 16.25/32 """
    handle, _, ticks = quote.strip().partition('-')
    if not ticks:
        return float(handle)
    if ticks.endswith('+'):
        frac = int(ticks[:-1]) + 0.5
    elif len(ticks) == 3:
        frac = int(ticks[:2]) + int(ticks[2]) / 8.0
    else:
        frac = int(ticks)
    return int(handle) + frac / 32.0


def to_32nds(price: float) -> str:
    """ Price to the nearest 1/4 of a 32nd, Treasury style: 99-16+ = 99 + 16.5/32, 99-162 = 99 + 16.25/32 """
    q = round(price * 128)  # quarters of 32nds
    handle, ticks = divmod(q, 128)
    whole, frac = divmod(ticks, 4)
    return f"{handle}-{whole:02d}" + {0: '', 1: '2', 2: '+', 3: '6'}[frac]


if __name__ == "__main__":
    # from sdevpy.utilities.scalendar import BDC
    # from sdevpy.utilities.sdaycount import DayCount
    from sdevpy.utilities.schedules import make_bond_scheduler
    from sdevpy.conventions.bondconventions import UST
    from sdevpy.rates.bonds import bond_flows, dirty_from_curve
    from sdevpy.calibration.rates import yieldcurve as ycrv

    valdate = dt.datetime(2026, 10, 26)
    settle = settlement_date(valdate, UST)
    print(f"Trade {valdate:%a %d-%b-%Y} -> settle {settle:%a %d-%b-%Y}")

    # Bill: 91 days, 4.00% discount rate
    bill = bill_flows(dt.date(2027, 1, 25), settle, valdate)
    p = price_from_discount_rate(bill, 0.04)
    print(f"Bill {bill_days(bill)}d: price {p:.6f}, back to rate {discount_rate_from_price(bill, p):.6%}")

    # 32nds
    for s in ['99-16', '99-16+', '99-162', '101-07', '100']:
        print(f"  {s:>7} = {parse_32nds(s):.6f}")

    # Note quoted in each style: all should give the same target dirty price
    ust = UST # LegConvention('6M', DayCount.ACTACT_ICMA, 'USGOV', bdc=BDC.F, adjust_accrual=False, eom=True)
    note = bond_flows(make_bond_scheduler(dt.date(2026, 8, 15), dt.date(2036, 8, 15), ust), settle, valdate)
    cpn, clean = 0.04125, parse_32nds('99-16+')
    dirty = clean + accrued_interest(note, cpn)
    ytm = yield_from_dirty(note, cpn, dirty)
    for qt, v in [(QuoteType.CLEAN, clean), (QuoteType.DIRTY, dirty), (QuoteType.YIELD, ytm)]:
        q = BondQuote(note, cpn, qt, v)
        tgt = target_dirty(q)
        print(f"  {qt.value:>8} {v:.8f} -> dirty {tgt:.8f}, weight {fit_weight(q, tgt):.4f} (1/DV01)")

    # Bill in the same framework: curve price = 100 DF(T)/DF(s), weight ~ 1/(P T 1e-4)
    curve = ycrv.InterpolatedYieldCurve(valdate=valdate, interp_var='zerorate', interp_type='linear')
    curve.set_data([dt.datetime(2027, 10, 26), dt.datetime(2028, 10, 26)], [np.exp(-0.04), np.exp(-0.08)])
    qb = BondQuote(bill, 0.0, QuoteType.DISCOUNT, 0.04)
    print(f"Bill target {target_dirty(qb):.6f}, curve {dirty_from_curve(bill, 0.0, curve.discount_float):.6f}, "
          f"weight {fit_weight(qb, target_dirty(qb)):.4f}")
