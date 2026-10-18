import datetime as dt
from sdevpy.utilities import scalendar


############################# TODO ################################################################
# * Define a cash-flow schedule class with acc start, end, pay and accrual
# * Update swap leg scheduling functions
# * Check if make_periods is really needed and why
# * A bond instrument has an ID (typically the ISIN) and a schedule
# * The schedule might be read from static data or generated if it's not available
# * The bond curve has a collection of bond instruments
# * The calibrated bond curve might be the YieldCurve object?
# * Need a pricing function that calculates the bond price given the schedule and the YieldCurve
# * Need to define the target function and the optimizer


def fixed_bond_schedule(trade_date: dt.datetime, maturity: str|dt.datetime, freq: str, calendar: str,
                        spot_lag: str='2D'):
    """ Generate bond schedule, i.e. list of: acc. start, acc. end, pay, accrual """
    cdr = scalendar.make_calendar(calendar)
    cdr.make_schedule(trade_date, maturity, freq)


class BondGenerator:
    def __init__(self):
        self.trade_date = ''


    def make_schedule(self):
        """ Make bond schedule """
        return 0


if __name__ == "__main__":
    # Define valuation date, currency
    # Define a list of pillars which each of a generator
    # Generators have IDs and correspond to conventions?
    #
    a = 0
    pillars = []

