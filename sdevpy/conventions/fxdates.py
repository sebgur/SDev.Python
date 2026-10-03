import datetime as dt
from sdevpy.utilities import dates as dts
from sdevpy.utilities.scalendar import make_calendar, BDC, to_eom, is_last_business_day_of_month


FX_SPOT_LAG_OVERRIDES = {
    frozenset({'USD', 'CAD'}): 1,
    frozenset({'USD', 'KZT'}): 1,
    frozenset({'USD', 'PHP'}): 1,
    frozenset({'USD', 'TRY'}): 1,
    frozenset({'CAD', 'KZT'}): 1,
    frozenset({'CAD', 'PHP'}): 1,
    frozenset({'CAD', 'TRY'}): 1,
    frozenset({'KZT', 'PHP'}): 1,
    frozenset({'KZT', 'TRY'}): 1,
    frozenset({'PHP', 'TRY'}): 1
}
DEFAULT_SPOT_LAG_DAYS = 2

# Note: if we want to generate the FX forward date (delivery, whatever we call it),
# we should do so using directly the fx_pillar_date() method in this file.
# That is: FX Forward(delivery = 1Y) => fx_pillar_date(valdate, "1Y").

def fx_spot_lag(forccy: str, domccy: str) -> int:
    """ Business days from trade date to spot, per FX market convention """
    return FX_SPOT_LAG_OVERRIDES.get(frozenset({forccy, domccy}), DEFAULT_SPOT_LAG_DAYS)


def fx_spot_date(date: dt.datetime, forccy: str, domccy: str) -> dt.datetime:
    """ USD pairs walk the lag on the joint currency-pair calendar. Crosses (neither leg USD) only need the first
        hop good in the pair's own centers, then require USD too, since crosses settle via two USD legs. """
    if forccy == domccy:
        raise ValueError(f"Foreign and domestic currency are the same: {forccy}")

    lag = fx_spot_lag(forccy, domccy)
    pair_cal = make_calendar(f"{forccy},{domccy}")

    if 'USD' in (forccy, domccy):
        return pair_cal.add_business_days(date, lag)

    settle_cal = make_calendar(f"{forccy},{domccy},USD")
    shifted_date = pair_cal.add_business_days(date, 1)
    return settle_cal.add_business_days(shifted_date, max(lag - 1, 0))
    # return settle_cal.add_business_days(shifted_date, lag - 1) if lag > 1 else shifted_date


def fx_pillar_date(valdate: dt.datetime, tenor_str: str, forccy: str, domccy: str,
                   convention: BDC=BDC.MF) -> dt.datetime:
    """ FX pillar date for a tenor, per standard market convention.
        ON/TN/SN settle directly off valdate (T+1/T+2/T+3 good business days in the pair's own calendar), not off spot.
        Every other tenor rolls off the spot date, which already carries fx_spot_date's settlement-lag conventions,
        including the USD overlay for crosses. By the tenor's calendar period, then business-day-adjusted in the pair's
        own calendar (no USD overlay here: the expiry/exercise date only needs to be good in the two option currencies'
        own centers, unlike settlement, which needs the correspondent-bank USD legs).
        End-of-month rule: if the spot date is the last good business day of its month, a whole-month/year tenor
        rolls to the last good business day of the target month too, rather than preserving day-of-month. """
    upper = tenor_str.upper()
    pair_cal = make_calendar(f"{forccy},{domccy}")

    if upper in ('ON', 'TN', 'SN'):
        n_days = {'ON': 1, 'TN': 2, 'SN': 3}[upper]
        raw = pair_cal.add_business_days(valdate, n_days)
    else:
        spot = fx_spot_date(valdate, forccy, domccy)
        rd = dts.period(tenor_str)
        raw = spot + rd
        if rd.days == 0 and is_last_business_day_of_month(pair_cal, spot):
            raw = to_eom(raw)
        # raw = spot + dts.period(tenor_str)

    return pair_cal.adjust(raw, convention)
