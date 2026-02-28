from sdevpy.montecarlo.payoffs.basic import *
from sdevpy.montecarlo.payoffs.vanillas import string_to_optiontype, vanilla_option, VanillaOptionPayoff


class WorstOfBarrier(Payoff):
    """ Not doing by algebra yet. Will need implementation of barrier monitoring first. """
    def __init__(self, names, strike, optiontype, barrier):
        super().__init__()
        self.names = names
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)
        self.barrier = barrier

    def evaluate(self, paths):
        spot_all = self.paths_for_all(paths)

        # Monitor barrier
        min_path = spot_all.min(axis=2)  # Worst asset at each time
        knocked = (min_path < self.barrier).any(axis=1)  # Knocked indicator

        # Payoff at expiry
        spot_all_at_exp = spot_all[:, -1, :]
        worst_at_exp = spot_all_at_exp.min(axis=1)
        payoff = vanilla_option(worst_at_exp, self.strike, self.optiontype)

        # Apply barrier
        payoff[knocked] = 0

        return payoff

    def set_nameindexes(self, names):
        self.set_multiindexes(names)

    # def set_valuation_date(self, valdate):
    #     for subpayoff in self.subpayoffs:
    #         subpayoff.set_valuation_date(valdate)

    #     # Gather event dates from subpayoofs
    #     self.eventdates = list_eventdates(self.subpayoffs)

    # def set_eventindexes(self, evendates):
    #     for subpayoff in self.subpayoffs:
    #         subpayoff.set_eventindexes(evendates)


def AsianOption(name, strike, optiontype, start, end, freq="1D", cdr="USD"):
    index = Average(name, start, end, freq, cdr)
    payoff = VanillaOptionPayoff(index, strike, optiontype)
    return payoff


def BasketOption(names, weights, strike, optiontype, expiry):
    spots = [Terminal(name, expiry) for name in names]
    basket = Basket(spots, weights)
    payoff = VanillaOptionPayoff(basket, strike, optiontype)
    return payoff


class Maximum(Payoff):
    """ Warning: work in progress. ToDo: document """
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_names([self.left, self.right])
        self.eventdates = list_eventdates([self.left, self.right])

    def evaluate(self, paths):
        return np.maximum(self.left.evaluate(paths), self.right.evaluate(paths))


class BarrierDown(Payoff):
    """ Warning: work in progress. ToDo: document """
    def __init__(self, asset_index, level):
        super().__init__()
        self.asset_index = asset_index
        self.level = level

    def evaluate(self, paths):
        breached = (paths[:, :, self.asset_index] < self.level).any(axis=1)
        return (~breached).astype(float)
