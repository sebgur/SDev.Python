from sdevpy.montecarlo.payoffs.basic import *
from sdevpy.montecarlo.payoffs.vanillas import string_to_optiontype, vanilla_option, VanillaOptionPayoff



class WorstOfBarrier(Payoff):
    def __init__(self, names, strike, optiontype, barrier):
        super().__init__(names)
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


def AsianOption(name, strike, optiontype):
    average = Average(name)
    payoff = VanillaOptionPayoff(average, strike, optiontype)
    return payoff


def BasketOption(names, weights, strike, optiontype):
    spots = [Terminal(name) for name in names]
    basket = Basket(spots, weights)
    payoff = VanillaOptionPayoff(basket, strike, optiontype)
    return payoff


class Maximum(Payoff):
    """ Warning: work in progress. ToDo: document """
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return np.maximum(self.left.evaluate(paths), self.right.evaluate(paths))


class BarrierDown(Payoff):
    """ Warning: work in progress. ToDo: document """
    def __init__(self, asset_index, level):
        self.asset_index = asset_index
        self.level = level

    def evaluate(self, paths):
        breached = (paths[:, :, self.asset_index] < self.level).any(axis=1)
        return (~breached).astype(float)

# class BasketOption(Payoff):
#     def __init__(self, names, weights, strike, optiontype):
#         super().__init__(names)
#         if len(weights) != len(names):
#             raise RuntimeError(f"Incompatible sizes between names and weights")

#         self.weights = weights
#         self.strike = strike
#         self.optiontype = string_to_optiontype(optiontype)

#     def evaluate(self, paths):
#         spot_all = self.paths_for_all(paths)
#         spot_all_at_exp = spot_all[:, -1, :]
#         basket = spot_all_at_exp @ self.weights
#         payoff = vanilla_option(basket, self.strike, self.optiontype)
#         return payoff
