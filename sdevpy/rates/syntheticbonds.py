import datetime as dt
import numpy as np
from sdevpy.utilities.scalendar import BDC, make_calendar
from sdevpy.utilities.sdaycount import DayCount
from sdevpy.utilities import schedules as sch
from sdevpy.rates import bonds
from sdevpy.rates import bondquotes as bq
from sdevpy.maths.specialfunctions import nelson_siegel


def sample_bond_universe(valdate: dt.datetime, ns_b0: float=0.052, ns_b1: float=-0.016, ns_b2: float=-0.022,
                         ns_tau: float=2.0, seed: int=42) -> dict:
    """ Generate a synthetic bond universe to test bond curve construction """
    # Fixed values
    calendar = 'USGOV'
    cal = make_calendar(calendar)
    settle = bq.settlement_date(valdate, calendar)
    conv = sch.LegConvention(frequency='6M', daycount=DayCount.ACTACT_ICMA, calendar=calendar, bdc=BDC.F,
                             adjust_accrual=False, eom=True)
    base_year = valdate.year
    rng = np.random.default_rng(seed)

    # Define target curve
    def target_zr(t: float) -> float:
        return nelson_siegel(t, ns_b0, ns_b1, ns_b2, ns_tau)

    def target_curve(t: float) -> float:
        """ Discount factor function """
        return np.exp(-target_zr(t) * t)

    # Generate quotes and labels
    quotes, labels = [], []

    # Bills: weekly maturities to 52 weeks, quoted on discount rate (3 decimals in %)
    for k in range(1, 53):
        mat = cal.adjust(settle + dt.timedelta(weeks=k), BDC.F)
        bill = bq.bill_flows(mat, settle, valdate)
        d = bq.discount_rate_from_price(bill, bonds.dirty_from_curve(bill, 0.0, target_curve))
        quotes.append(bq.BondQuote(bill, 0.0, bq.QuoteType.DISCOUNT, round(d, 5)))
        labels.append(('Bill', mat))

    # Notes/bonds: month-end maturities to 7Y, mid-month (Feb/May/Aug/Nov 15) to 30Y. Coupon near the par yield at
    # issue (1/8 increments), clean price quoted in 32nds with 1/2-tick noise
    for y in range(base_year + 1, base_year + 30):
        for m in range(1, 13):
            mats = []
            if y <= base_year + 7:
                mats.append(sch.to_date(np.datetime64(f"{y}-{m:02d}", 'M') + np.timedelta64(1, 'M')) \
                                                                           - dt.timedelta(days=1))
            if m in (2, 5, 8, 11):
                mats.append(dt.date(y, m, 15))
            for mat in mats:
                if mat <= settle + dt.timedelta(days=200):
                    continue
                tenor = 30 if mat.year > base_year + 20 else 10 if mat.day == 15 else 7
                iss = dt.date(mat.year - tenor, mat.month, min(mat.day, 28)) if mat.year - tenor >= 1995 else None
                iss = iss or dt.date(1995, mat.month, 15)
                bf_k = bonds.bond_flows(sch.make_bond_scheduler(iss, mat, conv), settle, valdate)
                cpn = max(0.00125, round((target_zr(tenor) + rng.normal(0, 0.012)) * 800) / 800)
                dirty_k = bonds.dirty_from_curve(bf_k, cpn, target_curve)
                clean_k = bq.parse_32nds(bq.to_32nds(dirty_k - bonds.accrued_interest(bf_k, cpn) + rng.normal(0, 1/64)))
                quotes.append(bq.BondQuote(bf_k, cpn, bq.QuoteType.CLEAN, clean_k))
                labels.append(('Note' if tenor < 20 else 'Bond', mat))

    return {'quotes': quotes, 'labels': labels, 'settlement': settle, 'target_zr': target_zr,
            'target_curve': target_curve, 'convention': conv}
