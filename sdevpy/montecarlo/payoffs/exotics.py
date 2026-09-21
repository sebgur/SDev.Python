import numpy as np
import datetime as dt
from sdevpy.montecarlo.payoffs.basic import Payoff, Average, Terminal, Basket
from sdevpy.montecarlo.payoffs.vanillas import string_to_optiontype, vanilla_option, make_vanilla_option_payoff
from sdevpy.market.provider import MarketDataProvider
from sdevpy.utilities.scalendar import make_schedule


class WorstOfBarrier(Payoff):
    """ Not doing by algebra yet. Will need implementation of barrier monitoring first.
        Known limitation: a trade struck in the past that has already knocked. Unlike Average, there's no fixings
        lookup for the barrier history, so monitoring starts at valdate only. ToDo: improve on that. """
    def __init__(self, names: list[str], date: dt.datetime, strike: float, optiontype, barrier,
                 freq: str="1D", cdr: str="USD"):
        super().__init__()
        self.names = names
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)
        self.barrier = barrier
        self.expiry = date
        self.freq = freq
        self.cdr = cdr
        self.expiry_idx = None
        self.monitor_idxs = None

    def evaluate(self, mkt_state: dict):
        paths = mkt_state.event_paths
        spot_all = self.paths_for_all(paths)
        monitored = spot_all[:, self.monitor_idxs, :] # This trade's own dates only

        # Monitor barrier
        min_path = monitored.min(axis=2) # Worst asset at each monitored time
        # min_path = spot_all.min(axis=2) # Worst asset at each time
        knocked = (min_path < self.barrier).any(axis=1) # Knocked indicator

        # Payoff at expiry
        spot_all_at_exp = spot_all[:, self.expiry_idx, :]
        worst_at_exp = spot_all_at_exp.min(axis=1)
        payoff = vanilla_option(worst_at_exp, self.strike, self.optiontype)

        # Apply barrier
        payoff[knocked] = 0

        return payoff

    def set_nameindexes(self, names):
        self.set_multiindexes(names)

    def set_valuation_date(self, valdate, md: MarketDataProvider):
        if self.expiry < valdate:
            raise ValueError("Past trade found")

        # self.eventdates = [self.expiry]
        dates = make_schedule(self.cdr, valdate, self.expiry, self.freq)
        self.eventdates = sorted(set(dates) | {self.expiry}) # expiry always included

    def set_eventindexes(self, eventdates):
        self.monitor_idxs = []
        for date in self.eventdates:
            matches = np.where(eventdates == date)[0]
            if len(matches) == 0:
                raise ValueError(f"Date {date} not found in event date grid")
            self.monitor_idxs.append(matches[0])

        self.expiry_idx = self.monitor_idxs[-1]  # expiry is the last date
        # matches = np.where(eventdates == self.expiry)[0]
        # if len(matches) == 0:
        #     raise ValueError(f"Date {self.expiry} not found in event date grid")
        # self.expiry_idx = matches[0]


def make_asian_option(name: str, strike: float, optiontype: str, start: dt.datetime, end: dt.datetime,
                      freq: str="1D", cdr: str="USD"):
    """ Create Asian option payoff """
    index = Average(name, start, end, freq, cdr)
    payoff = make_vanilla_option_payoff(index, strike, optiontype)
    return payoff


def make_basket_option(names: list[str], weights: list[float], strike: float, optiontype: str, expiry: dt.datetime):
    """ Create Basket option payoff """
    spots = [Terminal(name, expiry) for name in names]
    basket = Basket(spots, weights)
    payoff = make_vanilla_option_payoff(basket, strike, optiontype)
    return payoff
