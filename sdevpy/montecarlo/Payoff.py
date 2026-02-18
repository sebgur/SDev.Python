import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class Payoff(ABC):
    def __init__(self, names):
        self.names = names
        self.name_idxs = None
        self.name_dic = None

    @abstractmethod
    def build(self, paths):
        pass

    def paths_for_index(self, paths, name_idx):
        return paths[:, :, self.name_idxs[name_idx]]

    def paths_for_name(self, paths, name):
        return paths[:, :, self.name_dic[name]]

    def paths_for_all(self, paths):
        return paths[:, :, self.name_idxs]

    def set_pathindexes(self, pathnames):
        # Find path index for each name
        self.name_idxs = []
        self.name_dic = {}
        for name in self.names:
            try:
                idx = pathnames.index(name)
                self.name_idxs.append(idx)
                self.name_dic[name] = idx
            except Exception as e:
                raise ValueError(f"Could not find name {name} in path names: {str(e)}")

        # Check sizes
        if len(self.name_idxs) != len(self.names):
            raise ValueError(f"Incompatible sizes between names and path indexes")

        if len(self.name_dic.keys()) != len(self.names):
            raise ValueError(f"Incompatible sizes between names and name dictionary")


class WorstOfBarrier(Payoff):
    def __init__(self, names, strike, optiontype, barrier):
        super().__init__(names)
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)
        self.barrier = barrier

    def build(self, paths):
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

    def build(self, paths):
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

    def build(self, paths):
        spot_all = self.paths_for_all(paths)
        spot_all_at_exp = spot_all[:, -1, :]
        basket = spot_all_at_exp @ self.weights
        payoff = vanilla_option(basket, self.strike, self.optiontype)
        return payoff


class VanillaOptionType(Enum):
    CALL = 0
    PUT = 1
    STRADDLE = 2


class VanillaOption(Payoff):
    def __init__(self, name, strike, optiontype):
        super().__init__([name])
        self.name = name
        self.strike = strike
        self.optiontype = string_to_optiontype(optiontype)

    def build(self, paths):
        spot_path = self.paths_for_name(paths, self.name)  # self.paths_by_index(paths, 0)
        spot_at_exp = spot_path[:, -1] # Pick last time point for expiry
        payoff = vanilla_option(spot_at_exp, self.strike, self.optiontype)
        return payoff


def vanilla_option(spot, strike, optiontype):
    match optiontype:
        case VanillaOptionType.CALL: payoff = np.maximum(spot - strike, 0.0)
        case VanillaOptionType.PUT: payoff = np.maximum(strike - spot, 0.0)
        case VanillaOptionType.STRADDLE: payoff = np.abs(spot - strike)
        case _: raise RuntimeError(f"Invalid option type")

    return payoff


def string_to_optiontype(s):
    match s.lower():
        case 'call': return VanillaOptionType.CALL
        case 'put': return VanillaOption.PUT
        case 'straddle': return VanillaOption.STRADDLE
        case _: raise RuntimeError(f"Invalid option type: {s}")


if __name__ == "__main__":
    payoff = VanillaOption('SPX', 100, 'call')
    print(payoff)
