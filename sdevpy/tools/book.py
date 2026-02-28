import numpy as np
from sdevpy.montecarlo.payoffs.basic import list_names, list_eventdates
from sdevpy.tools.utils import isiterable


class Book:
    def __init__(self, trades=[], csa_curve_id="USD.SOFR.1D"):
        self.clear_trades()
        self.add_trades(trades)
        self.csa_curve_id = csa_curve_id

    def add_trades(self, trades):
        """ A single trade can be added, but it is more efficient to add trades
            by arrays """
        if isiterable(trades):
            self.trades.extend(trades)
            self.instruments.extend([t.instrument for t in trades])
        else:
            self.trades.append(trades)
            self.instruments.append(trades.instrument)

    def clear_trades(self):
        self.trades, self.instruments, self.names = [], [], None
        self.eventdates = None

    def set_nameindexes(self):
        self.names = list_names(self.instruments)
        for instr in self.instruments:
            instr.set_nameindexes(self.names)

        return self.names

    def set_valuation_date(self, valdate):
        # Set valuation date for each instrument. Among other things,
        # this sets the event dates in the instruments.
        for instr in self.instruments:
            instr.set_valuation_date(valdate)

        # Gather and merge event dates from all instruments
        all_dates = list_eventdates(self.instruments)
        live_dates = [d for d in all_dates if d >= valdate]
        self.eventdates = np.asarray(live_dates)

        # Now that we have the live event dates for the whole book, we can
        # set the indexes of the instrument event dates relative to the
        # total book's event dates
        for instr in self.instruments:
            instr.set_eventindexes(self.eventdates)
