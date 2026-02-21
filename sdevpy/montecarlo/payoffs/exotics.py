from sdevpy.montecarlo.payoffs.payoffs import Payoff



class Maximum(Payoff):
    """ ToDo: document """
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return np.maximum(self.left.evaluate(paths), self.right.evaluate(paths))


class BarrierDown(Payoff):
    """ ToDo: document """
    def __init__(self, asset_index, level):
        self.asset_index = asset_index
        self.level = level

    def evaluate(self, paths):
        breached = (paths[:, :, self.asset_index] < self.level).any(axis=1)
        return (~breached).astype(float)


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


class AsianOption(Payoff):
    def __init__(self, name, strike, optiontype):
        super().__init__([name])
        self.name = name
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)

    def evaluate(self, paths):
        spot_path = self.paths_for_index(paths, 0)  # Could use name too
        spot_average = spot_path.mean(axis=1)
        payoff = vanilla_option(spot_average, self.strike, self.optiontype)
        return payoff


class BasketOption(Payoff):
    def __init__(self, names, weights, strike, optiontype):
        super().__init__(names)
        if len(weights) != len(names):
            raise RuntimeError(f"Incompatible sizes between names and weights")

        self.weights = weights
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)

    def evaluate(self, paths):
        spot_all = self.paths_for_all(paths)
        spot_all_at_exp = spot_all[:, -1, :]
        basket = spot_all_at_exp @ self.weights
        payoff = vanilla_option(basket, self.strike, self.optiontype)
        return payoff
