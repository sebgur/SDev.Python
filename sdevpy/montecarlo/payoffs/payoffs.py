import numpy as np
from abc import ABC, abstractmethod
from enum import Enum


class Payoff(ABC):
    def __init__(self, names):
        self.names = names
        self.name_idxs = None
        self.name_dic = None

    @abstractmethod
    def evaluate(self, paths):
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

    #### Algebra ##################################################################################

    def __add__(self, other):
        return Add(self, ensure_node(other))

    def __sub__(self, other):
        return Sub(self, ensure_node(other))

    def __mul__(self, other):
        return Mul(self, ensure_node(other))

    def __truediv__(self, other):
        return Div(self, ensure_node(other))

    def __neg__(self):
        return Neg(self)


def ensure_node(x):
    if isinstance(x, Node):
        return x
    return Constant(x)


########### Primitives ##################################################################

class Constant(Payoff):
    def __init__(self, value):
        self.value = value

    def evaluate(self, paths):
        return np.full(paths.shape[0], self.value)


class Terminal(Payoff):
    """ Value of the asset on the last date of the time grid """
    def __init__(self, asset_index):
        self.asset_index = asset_index

    def evaluate(self, paths):
        return paths[:, -1, self.asset_index]


class Average(Payoff):
    """ Average value of the asset over time """
    def __init__(self, asset_index):
        self.asset_index = asset_index

    def evaluate(self, paths):
        return paths[:, :, self.asset_index].mean(axis=1)


class Max(Payoff):
    """ ToDo: document """
    def __init__(self, nodes):
        self.nodes = nodes

    def evaluate(self, paths):
        values = [node.evaluate(paths) for node in self.nodes]
        return np.max(np.column_stack(values), axis=1)


class Min(Payoff):
    """ ToDo: document """
    def __init__(self, nodes):
        self.nodes = nodes

    def evaluate(self, paths):
        values = [node.evaluate(paths) for node in self.nodes]
        return np.min(np.column_stack(values), axis=1)


########### Arithmetic Nodes ############################################################

class Add(Payoff):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return self.left.evaluate(paths) + self.right.evaluate(paths)


class Sub(Payoff):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return self.left.evaluate(paths) - self.right.evaluate(paths)


class Mul(Payoff):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return self.left.evaluate(paths) * self.right.evaluate(paths)


class Div(Payoff):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return self.left.evaluate(paths) / self.right.evaluate(paths)


class Neg(Payoff):
    def __init__(self, old):
        self.old = old

    def evaluate(self, paths):
        return -self.old.evaluate(paths)



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

    def evaluate(self, paths):
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
