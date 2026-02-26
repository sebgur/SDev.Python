from sdevpy.montecarlo.payoffs.basic import list_names
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

        self.names = list_names(self.instruments)

        # Set indexes
        for instr in self.instruments:
            instr.set_nameindexes(self.names)

    def clear_trades(self):
        self.trades, self.instruments, self.names = [], [], []
