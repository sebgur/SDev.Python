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

    def set_nameindexes(self):
        self.names = list_names(self.instruments)
        for instr in self.instruments:
            instr.set_nameindexes(self.names)

    def get_eventdates(self, valdate):
        """ Gather the event dates of each instrument that are equal to or later than the
            valuation date. Return the list of unique and ordered entries. """
        all_dates = list_eventdates(self.instruments)
        current_dates = [d for d in all_dates if d >= valdate]
        return np.asarray(current_dates)
