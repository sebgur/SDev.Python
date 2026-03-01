import numpy as np
import datetime as dt
from abc import ABC, abstractmethod
from sdevpy.tools.scalendar import make_schedule


def list_eventdates(payoffs):
    """ List the event dates behind the payoffs. Duplicates are removed and the
        result is ordered. """
    eventdates = []
    for payoff in payoffs:
        eventdates.extend(payoff.eventdates)

    eventdates = sorted(set(eventdates))
    return np.asarray(eventdates)


def list_names(payoffs):
    """ List the names behind the payoffs. Duplicates are removed and the result
        is ordered to avoid noise due to re-ordering of the random numbers depending
        on the order in which the trades are listed in the book. """
    names = []
    for payoff in payoffs:
        new_names = payoff.names
        if new_names is not None:
            names.extend(new_names)

    names = sorted(set(names))
    # names = names if len(names) != 0 else []
    return names


# market_state = {
#     "paths": paths,                        # full grid
#     "event_spots": event_spots,            # interpolated at event dates
#     "terminal_spots": paths[:, -1, :],     # shortcut
#     "discount_factors": df_curve,          # deterministic curve
#     "n_paths": n_paths
# }

class Trade:
    def __init__(self, instrument, **kwargs):
        self.instrument = instrument
        self.notional = kwargs.get('notional', 1.0)
        self.name = kwargs.get('name', '')


class Payoff(ABC):
    def __init__(self):
        self.names = None
        self.name_idxs = None
        self.name_dic = None
        self.eventdates = []

    @abstractmethod
    def generate_cashflows(self, paths):
        pass

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
        pass  # Do nothing in base

    def set_eventindexes(self, eventdates):
        pass  # Do nothing in base

    def set_valuation_date(self, valdate):
        pass  # Do nothing in base

    def set_multiindexes(self, enginenames):
        # Find path index for each name
        self.name_idxs = []
        self.name_dic = {}
        if self.names is not None:
            for name in self.names:
                try:
                    idx = enginenames.index(name)
                    self.name_idxs.append(idx)
                    self.name_dic[name] = idx
                except Exception as e:
                    raise ValueError(f"Could not find name {name} in path names: {str(e)}")

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

    def __radd__(self, other): # Otherwise doesn't work when starting with constant
        return Add(ensure_payoff(other), self)

    def __rsub__(self, other): # Otherwise doesn't work when starting with constant
        return Sub(ensure_payoff(other), self)

    def __rmul__(self, other): # Otherwise doesn't work when starting with constant
        return Mul(ensure_payoff(other), self)

    def __rtruediv__(self, other): # Otherwise doesn't work when starting with constant
        return Div(ensure_payoff(other), self)


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
        # print(f"Constant: {payoff.shape}")
        return payoff


class Terminal(Payoff):
    """ Value of the asset on the last date of the time grid """
    def __init__(self, name, date):
        super().__init__()
        self.names = [name]
        self.name = name
        self.name_idx = None
        self.expiry = date
        self.expiry_idx = None

    def evaluate(self, paths):
        # print(f"TPath shape: {paths.shape}")
        # print(f"expiry idx: {self.expiry_idx}")
        spot_at_exp = paths[:, self.expiry_idx, self.name_idx]
        # spot_at_exp = paths[:, -1, self.name_idx]
        # print(f"spot at exp: {spot_at_exp.shape}")
        payoff = spot_at_exp
        # print(f"Terminal: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        try:
            self.name_idx = names.index(self.name)
        except Exception as e:
            self.name_idx = None
            raise ValueError(f"Could not find name {name} in path names: {str(e)}")

    def set_valuation_date(self, valdate):
        if self.expiry < valdate:
            raise RuntimeError("Past trade found")

        self.eventdates = [self.expiry]

    def set_eventindexes(self, eventdates):
        self.expiry_idx = np.where(eventdates == self.expiry)[0][0]


class Average(Payoff):
    """ Average value of the asset over time """
    def __init__(self, name, start, end, freq="1D", cdr="USD"):
        super().__init__()
        self.names = [name]
        self.name = name
        self.name_idx, self.averageidxs = None, None
        self.start = start
        self.end = end
        self.alldates = make_schedule(cdr, self.start, self.end, freq)
        self.current_sum = None
        self.n_dates = len(self.alldates)

    def evaluate(self, paths):
        # print(f"APath: {paths.shape}")
        # print(f"Aidx: {self.averageidxs}")
        return paths[:, self.averageidxs, self.name_idx].mean(axis=1)
        # return paths[:, :, self.name_idx].mean(axis=1)

    def set_nameindexes(self, names):
        try:
            self.name_idx = names.index(self.name)
        except Exception as e:
            self.name_idx = None
            raise ValueError(f"Could not find name {name} in path names: {str(e)}")

    def set_valuation_date(self, valdate):
        # Calculate current sum using fixings up to the day before valdate
        # For days from and including vadate, collect the date as event date
        self.current_sum = 0.0
        self.eventdates = []
        for date in self.alldates:
            if date < valdate:
                fixing = 0.0  # ToDo: get from fixing data
                self.current_sum += fixing
            else:
                self.eventdates.append(date)

    def set_eventindexes(self, eventdates):
        self.averageidxs = []
        for date in self.eventdates:
            self.averageidxs.append(np.where(eventdates == date)[0][0])


class Max(Payoff):
    """ Max of the payoffs specified in the input list """
    def __init__(self, subpayoffs):
        super().__init__()
        subpayoffs = [ensure_payoff(node) for node in subpayoffs]
        self.subpayoffs = subpayoffs
        self.names = list_names(self.subpayoffs)
        # self.eventdates = list_eventdates(self.subpayoffs)

    def evaluate(self, paths):
        values = [subpayoff.evaluate(paths) for subpayoff in self.subpayoffs]
        # Create an array whose shape[0] is the number of paths and shape[1]
        # is the number of payoffs being maxed on each path. Then take the max
        # of the payoffs along the payoff direction (axis=1)
        payoff = np.max(np.column_stack(values), axis=1)
        # print(f"Max: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        for subpayoff in self.subpayoffs:
            subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for subpayoff in self.subpayoffs:
            subpayoff.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates(self.subpayoffs)

    def set_eventindexes(self, evendates):
        for subpayoff in self.subpayoffs:
            subpayoff.set_eventindexes(evendates)


class Min(Payoff):
    """ Min of the payoffs specified in the input list """
    def __init__(self, subpayoffs):
        super().__init__()
        subpayoffs = [ensure_payoff(node) for node in subpayoffs]
        self.subpayoffs = subpayoffs
        self.names = list_names(self.subpayoffs)
        # self.eventdates = list_eventdates(self.subpayoffs)

    def evaluate(self, paths):
        values = [subpayoff.evaluate(paths) for subpayoff in self.subpayoffs]
        # Create an array whose shape[0] is the number of paths and shape[1]
        # is the number of payoffs being maxed on each path. Then take the min
        # of the payoffs along the payoff direction (axis=1)
        payoff = np.min(np.column_stack(values), axis=1)
        # print(f"Min: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        for subpayoff in self.subpayoffs:
            subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for subpayoff in self.subpayoffs:
            subpayoff.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates(self.subpayoffs)

    def set_eventindexes(self, evendates):
        for subpayoff in self.subpayoffs:
            subpayoff.set_eventindexes(evendates)


class Abs(Payoff):
    """ Absolute value of the payoff """
    def __init__(self, subpayoff):
        super().__init__()
        self.subpayoff = subpayoff
        self.names = self.subpayoff.names
        # self.eventdates = self.subpayoffs.eventdates

    def evaluate(self, paths):
        old_path = self.subpayoff.evaluate(paths)
        payoff = np.abs(old_path)
        # print(f"Abs: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.subpayoff.set_valuation_date(valdate)
        self.eventdates = self.subpayoffs.evendates

    def set_eventindexes(self, evendates):
        self.subpayoff.set_eventindexes(evendates)


class Basket(Payoff):
    """ Linear combination of specified payoffs. We could have implemented it using
        the algebra, but the code below may help make the tree simpler """
    def __init__(self, subpayoffs, weights):
        super().__init__()
        self.subpayoffs = subpayoffs
        self.names = list_names(self.subpayoffs)
        # self.eventdates = list_eventdates(self.subpayoffs)
        self.weights = np.asarray(weights)
        if len(self.subpayoffs) != len(self.weights):
            raise RuntimeError("Incompatible sizes between sub-payoffs and weights")

    def evaluate(self, paths):
        sub_paths = np.asarray([p.evaluate(paths) for p in self.subpayoffs])
        # print(sub_paths.shape)
        # print(self.weights.shape)
        payoff = self.weights @ sub_paths
        # print(f"Basket: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        for subpayoff in self.subpayoffs:
            subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for subpayoff in self.subpayoffs:
            subpayoff.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates(self.subpayoffs)

    def set_eventindexes(self, evendates):
        for subpayoff in self.subpayoffs:
            subpayoff.set_eventindexes(evendates)


class WorstOf(Payoff):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def evaluate(self, paths):
        spot_all = self.paths_for_all(paths)
        spot_all_at_exp = spot_all[:, -1, :]
        worst_at_exp = spot_all_at_exp.min(axis=1)
        payoff = worst_at_exp
        # print(f"WorstOf: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.set_multiindexes(names)


########### Arithmetic Nodes ############################################################

class Add(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_names([self.left, self.right])
        # self.eventdates = list_eventdates([self.left, self.right])

    def evaluate(self, paths):
        payoff = self.left.evaluate(paths) + self.right.evaluate(paths)
        # print(f"Add: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates([self.left, self.right])

    def set_eventindexes(self, evendates):
        self.left.set_eventindexes(evendates)
        self.right.set_eventindexes(evendates)


class Sub(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_names([self.left, self.right])
        # self.eventdates = list_eventdates([self.left, self.right])

    def evaluate(self, paths):
        payoff = self.left.evaluate(paths) - self.right.evaluate(paths)
        # print(f"Sub: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates([self.left, self.right])

    def set_eventindexes(self, evendates):
        self.left.set_eventindexes(evendates)
        self.right.set_eventindexes(evendates)


class Mul(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_names([self.left, self.right])
        # self.eventdates = list_eventdates([self.left, self.right])

    def evaluate(self, paths):
        payoff = self.left.evaluate(paths) * self.right.evaluate(paths)
        # print(f"Mul: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates([self.left, self.right])

    def set_eventindexes(self, evendates):
        self.left.set_eventindexes(evendates)
        self.right.set_eventindexes(evendates)


class Div(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_names([self.left, self.right])
        # self.eventdates = list_eventdates([self.left, self.right])

    def evaluate(self, paths):
        return self.left.evaluate(paths) / self.right.evaluate(paths)
        # print(f"Div: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_eventdates([self.left, self.right])

    def set_eventindexes(self, evendates):
        self.left.set_eventindexes(evendates)
        self.right.set_eventindexes(evendates)


class Neg(Payoff):
    def __init__(self, old):
        super().__init__()
        self.old = old
        self.names = self.old.names
        # self.eventdates = self.old.eventdates

    def set_nameindexes(self, names):
        self.old.set_nameindexes(names)

    def evaluate(self, paths):
        payoff = -self.old.evaluate(paths)
        print(f"Neg: {payoff.shape}")
        return payoff

    def set_valuation_date(self, valdate):
        self.old.set_valuation_date(valdate)
        self.eventdates = self.old.eventdates

    def set_eventindexes(self, evendates):
        self.old.set_eventindexes(evendates)


#### Event Dates ##################################################################################


if __name__ == "__main__":
    from sdevpy.montecarlo.MonteCarloPricer import build_eventdate_interpolator, interpolate_paths
    # Discretization
    disc_times = np.asarray([0, 1, 2])
    disc_paths = np.asarray([
        [[100, 10],[110, 11],[120, 12]],
        [[100, 10],[90, 9],[80, 8]],
        [[100, 10],[150, 15],[200, 20]],
        [[100, 10],[70, 7],[60, 6]]])

    # Event dates
    event_times = np.asarray([0.0, 0.2, 1.0, 1.4, 2.0, 2.5])

    # Interpolation
    idx, w0, w1 = build_eventdate_interpolator(disc_times, event_times)
    int_paths = interpolate_paths(disc_paths, idx, w0, w1)

    # Display
    np.set_printoptions(suppress=True, precision=2)
    print(f"Number paths: {disc_paths.shape[0]}")
    print(f"Number disc. times: {len(disc_times)}")
    print(f"Number factors: {disc_paths.shape[2]}")
    print(f"Number event dates: {len(event_times)}")
    print(f"Disc. path shape: {disc_paths.shape}")
    print(disc_paths)
    print(f"Interp. path shape: {int_paths.shape}")
    print(int_paths)
