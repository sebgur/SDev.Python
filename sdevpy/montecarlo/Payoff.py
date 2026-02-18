import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class Payoff(ABC):
    def __init__(self, names):
        self.names = names
        self.name_idxs = None

    @abstractmethod
    def build(self, paths):
        pass
        # return self.payoff_function(paths)

    def set_pathindexes(self, pathnames):
        # Find path index for each name
        self.name_idxs = []
        for name in self.names:
            try:
                self.name_idxs.append(pathnames.index(name))
            except Exception as e:
                raise ValueError(f"Could not find name {name} in path names: {str(e)}")

        # Check sizes
        if len(self.names) != len(self.name_idxs):
            raise ValueError(f"Incompatible sizes between names and path indexes")


def callput_payoff(paths, name, strike, optiontype, names):
    ST = paths[:, -1, :]  # Pick last time point
    print(ST.shape)
    print(weights)
    payoff = np.maximum(ST - strike, 0)
    return payoff


def worst_of_barrier(paths, strike, barrier):
    min_path = paths.min(axis=2)  # worst asset at each time
    knocked = (min_path < barrier).any(axis=1)
    ST = paths[:, -1, :] # Pick last time point
    worst_final = ST.min(axis=1)

    payoff = np.maximum(worst_final - strike, 0)
    payoff[knocked] = 0
    return payoff


def asian_call_payoff(paths, strike):
    avg = paths.mean(axis=1)[:, 0]  # asset 0
    return np.maximum(avg - strike, 0)


def basket_call_payoff(paths, strike, weights):
    ST = paths[:, -1, :]
    basket = ST @ weights
    return np.maximum(basket - strike, 0)


class VanillaOptionType(Enum):
    CALL = 0
    PUT = 1
    STRADDLE = 2


class VanillaOption(Payoff):
    def __init__(self, name, strike, optiontype):
        super().__init__([name])
        self.strike = strike
        self.path_idx = None
        # self.name = name
        match optiontype.lower():
            case 'call': self.optiontype = VanillaOptionType.CALL
            case 'put': self.optiontype = VanillaOption.PUT
            case 'straddle': self.optiontype = VanillaOption.STRADDLE
            case _: raise RuntimeError(f"Invalid option type: {optiontype}")

    def build(self, paths):
        spot = paths[:, -1, self.name_idxs[0]]  # Pick last time point
        print(f"Index path shape: {spot.shape}")
        payoff = np.maximum(spot - self.strike, 0.0)
        print(f"Payoff shape: {payoff.shape}")
        return payoff

    # def set_pathnames(self, names):
    #     try:
    #         self.path_idx = names.index(self.name)
    #     except Exception as e:
    #         self.path_idx = None
    #         raise ValueError(f"Could not find name {self.name} in path names: {str(e)}")


if __name__ == "__main__":
    payoff = VanillaOption('SPX', 100, 'call')
    print(payoff)
