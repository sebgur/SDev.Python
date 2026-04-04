import numpy as np
from sdevpy.montecarlo.payoffs.basic import (
    Payoff, Average, Terminal, Basket,
    list_payoff_names, list_payoff_eventdates)
from sdevpy.montecarlo.payoffs.vanillas import string_to_optiontype, vanilla_option, make_vanilla_option_payoff


class WorstOfBarrier(Payoff):
    """ Not doing by algebra yet. Will need implementation of barrier monitoring first. """
    def __init__(self, names, date, strike, optiontype, barrier):
        super().__init__()
        self.names = names
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)
        self.barrier = barrier
        self.expiry = date
        self.expiry_idx = None

    def evaluate(self, mkt_state):
        paths = mkt_state.event_paths
        spot_all = self.paths_for_all(paths)

        # Monitor barrier
        min_path = spot_all.min(axis=2) # Worst asset at each time
        knocked = (min_path < self.barrier).any(axis=1) # Knocked indicator

        # Payoff at expiry
        spot_all_at_exp = spot_all[:, self.expiry_idx, :]
        # spot_all_at_exp = spot_all[:, -1, :]
        worst_at_exp = spot_all_at_exp.min(axis=1)
        payoff = vanilla_option(worst_at_exp, self.strike, self.optiontype)

        # Apply barrier
        payoff[knocked] = 0

        return payoff

    def set_nameindexes(self, names):
        self.set_multiindexes(names)

    def set_valuation_date(self, valdate):
        if self.expiry < valdate:
            raise RuntimeError("Past trade found")

        self.eventdates = [self.expiry]

    def set_eventindexes(self, eventdates):
        matches = np.where(eventdates == self.expiry)[0]
        if len(matches) == 0:
            raise ValueError(f"Date {self.expiry} not found in event date grid")
        self.expiry_idx = matches[0]


def make_asian_option(name, strike, optiontype, start, end, freq="1D", cdr="USD"):
    index = Average(name, start, end, freq, cdr)
    payoff = make_vanilla_option_payoff(index, strike, optiontype)
    return payoff


def make_basket_option(names, weights, strike, optiontype, expiry):
    spots = [Terminal(name, expiry) for name in names]
    basket = Basket(spots, weights)
    payoff = make_vanilla_option_payoff(basket, strike, optiontype)
    return payoff


class Maximum(Payoff):
    """ Warning: work in progress. ToDo: document """
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_payoff_names([self.left, self.right])
        self.eventdates = list_payoff_eventdates([self.left, self.right])

    def evaluate(self, mkt_state):
        return np.maximum(self.left.evaluate(mkt_state), self.right.evaluate(mkt_state))


class BarrierDown(Payoff):
    """ Warning: work in progress. ToDo: document """
    def __init__(self, asset_index, level):
        super().__init__()
        self.asset_index = asset_index
        self.level = level

    def evaluate(self, mkt_state):
        paths = mkt_state.event_paths
        breached = (paths[:, :, self.asset_index] < self.level).any(axis=1)
        return (~breached).astype(float)
