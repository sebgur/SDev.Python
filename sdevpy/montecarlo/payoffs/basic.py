import numpy as np
from abc import ABC, abstractmethod


class Payoff(ABC):
    def __init__(self, names=None):
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

    def set_nameindexes(self, names):
        pass # Do nothing in base

    # ToDo: review if still needed
    def set_pathindexes(self, pathnames):
        # Find path index for each name
        self.name_idxs = []
        self.name_dic = {}
        if self.names is not None:
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
        return Add(self, ensure_payoff(other))

    def __sub__(self, other):
        return Sub(self, ensure_payoff(other))

    def __mul__(self, other):
        return Mul(self, ensure_payoff(other))

    def __truediv__(self, other):
        return Div(self, ensure_payoff(other))

    def __neg__(self):
        return Neg(self)

    def __rsub__(self, other): # Otherwise doesn't work when starting with constant
        return Sub(ensure_payoff(other), self)

    def __rmul__(self, other): # Otherwise doesn't work when starting with constant
        return Mul(ensure_payoff(other), self)

    def __rtruediv__(self, other): # Otherwise doesn't work when starting with constant
        return Div(ensure_payoff(other), self)


def add_names(payoffs):
    names = []
    for payoff in payoffs:
        new_names = payoff.names
        if new_names is not None:
            names.extend(new_names)

    names = names if len(names) != 0 else None
    return names


def ensure_payoff(x):
    if isinstance(x, Payoff):
        return x
    return Constant(x)


########### Primitives ##################################################################

class Constant(Payoff):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def evaluate(self, paths):
        payoff = np.full(paths.shape[0], self.value)
        print(f"Constant: {payoff.shape}")
        return payoff


class Terminal(Payoff):
    """ Value of the asset on the last date of the time grid """
    def __init__(self, name): #asset_index):
        super().__init__([name])
        self.name = name
        self.name_idx = None

    def set_nameindexes(self, names):
        try:
            self.name_idx = names.index(self.name)
        except Exception as e:
            self.name_idx = None
            raise ValueError(f"Could not find name {name} in path names: {str(e)}")

    def evaluate(self, paths):
        spot_at_exp = paths[:, -1, self.name_idx]
        # print(spot_at_exp.shape)
        # spot_path = self.paths_for_name(paths, self.name)  # self.paths_by_index(paths, 0)
        # spot_at_exp = spot_path[:, -1] # Pick last time point for expiry
        # payoff = vanilla_option(spot_at_exp, self.strike, self.optiontype)
        payoff = spot_at_exp
        print(f"Terminal: {payoff.shape}")
        return payoff


class Average(Payoff):
    """ Average value of the asset over time """
    def __init__(self, asset_index):
        self.asset_index = asset_index

    def evaluate(self, paths):
        return paths[:, :, self.asset_index].mean(axis=1)


class Max(Payoff):
    """ Max of the payoffs specified in the input list """
    def __init__(self, nodes):
        nodes = [ensure_payoff(node) for node in nodes]
        self.nodes = nodes

    def set_nameindexes(self, names):
        for node in self.nodes:
            node.set_nameindexes(names)

    def evaluate(self, paths):
        values = [node.evaluate(paths) for node in self.nodes]
        # Create an array whose shape[0] is the number of paths and shape[1]
        # is the number of payoffs being maxed on each path. Then take the max
        # of the payoffs along the payoff direction (axis=1)
        payoff = np.max(np.column_stack(values), axis=1)
        print(f"Max: {payoff.shape}")
        return payoff


class Min(Payoff):
    """ Min of the payoffs specified in the input list """
    def __init__(self, nodes):
        self.nodes = nodes

    def set_nameindexes(self, names):
        for node in self.nodes:
            node.set_nameindexes(names)

    def evaluate(self, paths):
        values = [node.evaluate(paths) for node in self.nodes]
        # Create an array whose shape[0] is the number of paths and shape[1]
        # is the number of payoffs being maxed on each path. Then take the min
        # of the payoffs along the payoff direction (axis=1)
        payoff = np.min(np.column_stack(values), axis=1)
        print(f"Min: {payoff.shape}")
        return payoff


class Abs(Payoff):
    """ Absolute value of the payoff """
    def __init__(self, old):
        self.old = old

    def set_nameindexes(self, names):
        self.old.set_nameindexes(names)

    def evaluate(self, paths):
        old_path = self.old.evaluate(paths)
        payoff = np.abs(old_path)
        print(f"Abs: {payoff.shape}")
        return payoff


########### Arithmetic Nodes ############################################################

class Add(Payoff):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def evaluate(self, paths):
        return self.left.evaluate(paths) + self.right.evaluate(paths)


class Sub(Payoff):
    def __init__(self, left, right):
        # super().__init__()
        self.left = left
        self.right = right

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def evaluate(self, paths):
        payoff = self.left.evaluate(paths) - self.right.evaluate(paths)
        print(f"Sub: {payoff.shape}")
        return payoff


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

    def set_nameindexes(self, names):
        self.old.set_nameindexes(names)

    def evaluate(self, paths):
        payoff = -self.old.evaluate(paths)
        print(f"Neg: {payoff.shape}")
        return payoff

