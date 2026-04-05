import numpy as np
import datetime as dt
from abc import ABC, abstractmethod
from sdevpy.tools.scalendar import make_schedule
from sdevpy.tools.utils import hash
from sdevpy.market import fixings as fxgs


def list_payoff_eventdates(payoffs):
    """ List the event dates behind the payoffs. Duplicates are removed and the
        result is ordered. """
    eventdates = []
    for payoff in payoffs:
        eventdates.extend(payoff.eventdates)

    eventdates = sorted(set(eventdates))
    return np.asarray(eventdates)


def list_instrument_eventdates(instruments):
    """ List the event dates behind the instrument. Duplicates are removed and the
        result is ordered. """
    eventdates = []
    for instr in instruments:
        eventdates.extend(instr.eventdates())

    eventdates = sorted(set(eventdates))
    return np.asarray(eventdates)


def list_payoff_names(payoffs):
    """ List the names behind the payoffs. Duplicates are removed and the result
        is ordered to avoid noise due to re-ordering of the random numbers depending
        on the order in which the trades are listed in the book. """
    names = []
    for payoff in payoffs:
        new_names = payoff.names
        if new_names is not None:
            names.extend(new_names)

    names = sorted(set(names))
    return names

def list_instrument_names(instruments):
    """ List the names behind the instruments. Duplicates are removed and the result
        is ordered to avoid noise due to re-ordering of the random numbers depending
        on the order in which the trades are listed in the book. """
    names = []
    for instr in instruments:
        new_names = instr.names()
        if new_names is not None:
            names.extend(new_names)

    names = sorted(set(names))
    return names


class Instrument:
    def __init__(self, cashflow_legs, **kwargs):
        self.cashflow_legs = cashflow_legs
        self.id = kwargs.get('id', hash())

    def names(self):
        payoffs = []
        for leg in self.cashflow_legs:
            for cf in leg:
                payoffs.append(cf.payoff)

        return list_payoff_names(payoffs)

    def eventdates(self):
        payoffs = []
        for leg in self.cashflow_legs:
            for cf in leg:
                payoffs.append(cf.payoff)

        return list_payoff_eventdates(payoffs)

    def set_nameindexes(self, names):
        for leg in self.cashflow_legs:
            for cf in leg:
                cf.payoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for leg in self.cashflow_legs:
            for cf in leg:
                cf.payoff.set_valuation_date(valdate)

    def set_eventindexes(self, eventdates):
        for leg in self.cashflow_legs:
            for cf in leg:
                cf.payoff.set_eventindexes(eventdates)


class Trade:
    def __init__(self, instrument, **kwargs):
        self.instrument = instrument
        self.notional = kwargs.get('notional', 1.0)
        self.id = kwargs.get('id', hash())


class Payoff(ABC):
    def __init__(self):
        self.names = None
        self.name_idxs = None
        self.name_dic = None
        self.eventdates = []

    @abstractmethod
    def evaluate(self, mkt_state: dict):
        pass

    def paths_for_index(self, paths, name_idx):
        return paths[:, :, self.name_idxs[name_idx]]

    def paths_for_name(self, paths, name):
        return paths[:, :, self.name_dic[name]]

    def paths_for_all(self, paths):
        return paths[:, :, self.name_idxs]

    def set_nameindexes(self, names): # noqa: B027
        """ No-op default: override in subclasses when needed """
        pass

    def set_eventindexes(self, eventdates): # noqa: B027
        """ No-op default: override in subclasses when needed """
        pass

    def set_valuation_date(self, valdate): # noqa: B027
        """ No-op default: override in subclasses when needed """
        pass

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
                except ValueError as e:
                    raise ValueError(f"Could not find name {name} in path names: {str(e)}") from e

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

    def evaluate(self, mkt_state: dict):
        paths = mkt_state.event_paths
        payoff = np.full(paths.shape[0], self.value)
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

    def evaluate(self, mkt_state: dict):
        paths = mkt_state.event_paths
        spot_at_exp = paths[:, self.expiry_idx, self.name_idx]
        payoff = spot_at_exp
        return payoff

    def set_nameindexes(self, names):
        try:
            self.name_idx = names.index(self.name)
        except ValueError as e:
            raise ValueError(f"Could not find name {self.name} in path names: {str(e)}") from e

    def set_valuation_date(self, valdate):
        if self.expiry < valdate:
            raise RuntimeError("Past trade found")

        self.eventdates = [self.expiry]

    def set_eventindexes(self, eventdates):
        matches = np.where(eventdates == self.expiry)[0]
        if len(matches) == 0:
            raise ValueError(f"Date {self.expiry} not found in event date grid")
        self.expiry_idx = matches[0]


class Average(Payoff):
    """ Average value of the asset over time.
        ToDo: test with inception date in the past """
    def __init__(self, name, start, end, freq="1D", cdr="USD"):
        super().__init__()
        self.names = [name]
        self.name = name
        self.name_idx, self.averageidxs = None, None
        self.start = start
        self.end = end
        self.alldates = make_schedule(cdr, self.start, self.end, freq)
        self.current_sum = 0.0
        self.n_samples = len(self.alldates)

    def evaluate(self, mkt_state: dict):
        paths = mkt_state.event_paths
        new_sum = paths[:, self.averageidxs, self.name_idx].sum(axis=1)
        # average = new_sum / len(self.averageidxs)
        return (self.current_sum + new_sum) / self.n_samples

    def set_nameindexes(self, names):
        try:
            self.name_idx = names.index(self.name)
        except ValueError as e:
            raise ValueError(f"Could not find name {self.name} in path names: {str(e)}") from e

    def set_valuation_date(self, valdate):
        # Calculate current sum using fixings up to the day before valdate
        # For days from and including valdate, collect the date as event date
        self.eventdates = []
        hist_fixing_dates = []
        for date in self.alldates:
            if date < valdate:
                hist_fixing_dates.append(date)
            else:
                self.eventdates.append(date)

        # Fetch historical fixings
        hist_fixings = fxgs.get_fixings(self.name, hist_fixing_dates)

        # Calculate historical sum up to the day before valuation
        self.current_sum = np.asarray(hist_fixings).sum()


    def set_eventindexes(self, eventdates):
        self.averageidxs = []
        for date in self.eventdates:
            matches = np.where(eventdates == date)[0]
            if len(matches) == 0:
                raise ValueError(f"Date {date} not found in event date grid")
            self.averageidxs.append(matches[0])


class Max(Payoff):
    """ Max of the payoffs specified in the input list """
    def __init__(self, subpayoffs):
        super().__init__()
        subpayoffs = [ensure_payoff(node) for node in subpayoffs]
        self.subpayoffs = subpayoffs
        self.names = list_payoff_names(self.subpayoffs)

    def evaluate(self, mkt_state: dict):
        values = [subpayoff.evaluate(mkt_state) for subpayoff in self.subpayoffs]
        # Create an array whose shape[0] is the number of paths and shape[1]
        # is the number of payoffs being maxed on each path. Then take the max
        # of the payoffs along the payoff direction (axis=1)
        payoff = np.max(np.column_stack(values), axis=1)
        return payoff

    def set_nameindexes(self, names):
        for subpayoff in self.subpayoffs:
            subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for subpayoff in self.subpayoffs:
            subpayoff.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates(self.subpayoffs)

    def set_eventindexes(self, eventdates):
        for subpayoff in self.subpayoffs:
            subpayoff.set_eventindexes(eventdates)


class Min(Payoff):
    """ Min of the payoffs specified in the input list """
    def __init__(self, subpayoffs):
        super().__init__()
        subpayoffs = [ensure_payoff(node) for node in subpayoffs]
        self.subpayoffs = subpayoffs
        self.names = list_payoff_names(self.subpayoffs)

    def evaluate(self, mkt_state: dict):
        values = [subpayoff.evaluate(mkt_state) for subpayoff in self.subpayoffs]
        # Create an array whose shape[0] is the number of paths and shape[1]
        # is the number of payoffs being maxed on each path. Then take the min
        # of the payoffs along the payoff direction (axis=1)
        payoff = np.min(np.column_stack(values), axis=1)
        return payoff

    def set_nameindexes(self, names):
        for subpayoff in self.subpayoffs:
            subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for subpayoff in self.subpayoffs:
            subpayoff.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates(self.subpayoffs)

    def set_eventindexes(self, eventdates):
        for subpayoff in self.subpayoffs:
            subpayoff.set_eventindexes(eventdates)


class Abs(Payoff):
    """ Absolute value of the payoff """
    def __init__(self, subpayoff):
        super().__init__()
        self.subpayoff = subpayoff
        self.names = self.subpayoff.names

    def evaluate(self, mkt_state: dict):
        old_path = self.subpayoff.evaluate(mkt_state)
        payoff = np.abs(old_path)
        return payoff

    def set_nameindexes(self, names):
        self.subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.subpayoff.set_valuation_date(valdate)
        self.eventdates = self.subpayoff.eventdates

    def set_eventindexes(self, eventdates):
        self.subpayoff.set_eventindexes(eventdates)


class Basket(Payoff):
    """ Linear combination of specified payoffs. We could have implemented it using
        the algebra, but the code below may help make the tree simpler """
    def __init__(self, subpayoffs, weights):
        super().__init__()
        self.subpayoffs = subpayoffs
        self.names = list_payoff_names(self.subpayoffs)
        self.weights = np.asarray(weights)
        if len(self.subpayoffs) != len(self.weights):
            raise ValueError("Incompatible sizes between sub-payoffs and weights")

    def evaluate(self, mkt_state: dict):
        sub_paths = np.asarray([p.evaluate(mkt_state) for p in self.subpayoffs])
        payoff = self.weights @ sub_paths
        return payoff

    def set_nameindexes(self, names):
        for subpayoff in self.subpayoffs:
            subpayoff.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        for subpayoff in self.subpayoffs:
            subpayoff.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates(self.subpayoffs)

    def set_eventindexes(self, eventdates):
        for subpayoff in self.subpayoffs:
            subpayoff.set_eventindexes(eventdates)


class WorstOf(Payoff):
    def __init__(self, names: list[str], date: dt.datetime):
        super().__init__()
        self.names = names
        self.expiry = date
        self.expiry_idx = None

    def evaluate(self, mkt_state: dict):
        paths = mkt_state.event_paths
        spot_all = self.paths_for_all(paths)
        spot_all_at_exp = spot_all[:, self.expiry_idx, :]
        worst_at_exp = spot_all_at_exp.min(axis=1)
        payoff = worst_at_exp
        return payoff

    def set_nameindexes(self, names):
        self.set_multiindexes(names)

    def set_valuation_date(self, valdate):
        if self.expiry < valdate:
            raise ValueError("Past trade found")

        self.eventdates = [self.expiry]

    def set_eventindexes(self, eventdates):
        matches = np.where(eventdates == self.expiry)[0]
        if len(matches) == 0:
            raise ValueError(f"Date {self.expiry} not found in event date grid")
        self.expiry_idx = matches[0]


class Variance(Payoff):
    """ Variance value of the asset over time. It calculates

        variance = 10000 x 252 / num_returns * sum log(return)^2

        That is, we calculate the (non-centered, simply summed) variance from trade start
        to trade end, then normalize to 1 day by dividing by the total number of returns
        from start to end, then multiply by 252 to express it in annual terms, then
        multiply by 10000 to scale to standard quotation terms.  """
    def __init__(self, name, start, end, freq="1D", cdr="USD"):
        super().__init__()
        self.names = [name]
        self.name = name
        self.name_idx, self.varidxs = None, None
        self.start = start
        self.end = end
        self.alldates = make_schedule(cdr, self.start, self.end, freq)
        self.current_sum = 0.0
        self.current_fixing = None
        self.n_dates = len(self.alldates)
        self.n_returns = len(self.alldates) - 1
        self.scaling = 10000 * 252

    def evaluate(self, mkt_state: dict):
        paths = mkt_state.event_paths
        spot_paths = paths[:, self.varidxs, self.name_idx]
        # Historical variance
        var_sum = self.current_sum
        # Add current increment
        if self.current_fixing is None:
            raise ValueError(f"Fixing not set for variance on {self.name}")

        var_sum = var_sum + np.power(np.log(spot_paths[:, 0] / self.current_fixing), 2)
        # Add forward variance
        log_returns = np.diff(np.log(np.asarray(spot_paths)))
        log_returns2 = np.power(log_returns, 2)
        var_sum = var_sum + log_returns2.sum(axis=1)
        return self.scaling * var_sum / self.n_returns

    def set_nameindexes(self, names):
        try:
            self.name_idx = names.index(self.name)
        except ValueError as e:
            self.name_idx = None
            raise ValueError(f"Could not find name {self.name} in path names: {str(e)}") from e

    def set_valuation_date(self, valdate):
        # Calculate current variance using fixings up to the day before valdate
        # For days from and including valdate, collect the date as event date
        self.eventdates = []
        hist_fixing_dates = []
        for date in self.alldates:
            if date < valdate:
                hist_fixing_dates.append(date)
            else:
                self.eventdates.append(date)

        # Fetch historical fixings
        hist_fixings = fxgs.get_fixings(self.name, hist_fixing_dates)

        # Calculate historical variance up to the day before valuation
        self.current_sum = 0.0
        if hist_fixings is not None and len(hist_fixings) > 1:
            log_returns = np.diff(np.log(np.asarray(hist_fixings)))
            self.current_sum = np.power(log_returns, 2).sum()

        if hist_fixings is not None and len(hist_fixings) >= 1:
            self.current_fixing = hist_fixings[-1]


    def set_eventindexes(self, eventdates):
        self.varidxs = []
        for date in self.eventdates:
            matches = np.where(eventdates == date)[0]
            if len(matches) == 0:
                raise ValueError(f"Date {date} not found in event date grid")
            self.varidxs.append(matches[0])

########### Arithmetic Nodes ############################################################

class Add(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_payoff_names([self.left, self.right])

    def evaluate(self, mkt_state: dict):
        payoff = self.left.evaluate(mkt_state) + self.right.evaluate(mkt_state)
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates([self.left, self.right])

    def set_eventindexes(self, eventdates):
        self.left.set_eventindexes(eventdates)
        self.right.set_eventindexes(eventdates)


class Sub(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_payoff_names([self.left, self.right])

    def evaluate(self, mkt_state: dict):
        payoff = self.left.evaluate(mkt_state) - self.right.evaluate(mkt_state)
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates([self.left, self.right])

    def set_eventindexes(self, eventdates):
        self.left.set_eventindexes(eventdates)
        self.right.set_eventindexes(eventdates)


class Mul(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_payoff_names([self.left, self.right])

    def evaluate(self, mkt_state: dict):
        payoff = self.left.evaluate(mkt_state) * self.right.evaluate(mkt_state)
        # print(f"Mul: {payoff.shape}")
        return payoff

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates([self.left, self.right])

    def set_eventindexes(self, eventdates):
        self.left.set_eventindexes(eventdates)
        self.right.set_eventindexes(eventdates)


class Div(Payoff):
    def __init__(self, left, right):
        super().__init__()
        self.left = left
        self.right = right
        self.names = list_payoff_names([self.left, self.right])

    def evaluate(self, mkt_state: dict):
        return self.left.evaluate(mkt_state) / self.right.evaluate(mkt_state)

    def set_nameindexes(self, names):
        self.left.set_nameindexes(names)
        self.right.set_nameindexes(names)

    def set_valuation_date(self, valdate):
        self.left.set_valuation_date(valdate)
        self.right.set_valuation_date(valdate)

        # Gather event dates from subpayoofs
        self.eventdates = list_payoff_eventdates([self.left, self.right])

    def set_eventindexes(self, eventdates):
        self.left.set_eventindexes(eventdates)
        self.right.set_eventindexes(eventdates)


class Neg(Payoff):
    def __init__(self, old):
        super().__init__()
        self.old = old
        self.names = self.old.names

    def set_nameindexes(self, names):
        self.old.set_nameindexes(names)

    def evaluate(self, mkt_state: dict):
        payoff = -self.old.evaluate(mkt_state)
        return payoff

    def set_valuation_date(self, valdate):
        self.old.set_valuation_date(valdate)
        self.eventdates = self.old.eventdates

    def set_eventindexes(self, eventdates):
        self.old.set_eventindexes(eventdates)


if __name__ == "__main__":
    from sdevpy.montecarlo.mcpricer import path_interp_coeffs, interp_paths
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
    idx, w0, w1 = path_interp_coeffs(disc_times, event_times)
    int_paths = interp_paths(disc_paths, idx, w0, w1)

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
