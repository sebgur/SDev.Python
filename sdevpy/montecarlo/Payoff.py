import numpy as np


def callput_payoff(paths, strikes, callputs, weights):
    return 0


def worst_of_barrier(paths, strike, barrier):
    min_path = paths.min(axis=2)  # worst asset at each time
    knocked = (min_path < barrier).any(axis=1)
    ST = paths[:, -1, :]
    worst_final = ST.min(axis=1)

    payoff = np.maximum(worst_final - strike, 0)
    payoff[knocked] = 0
    return payoff


def asian_call_payoff(paths, strike):
    avg = paths.mean(axis=1)[:, 0]  # asset 0
    return np.maximum(avg - strike, 0)


def basket_call_payoff(paths, weights, strike):
    ST = paths[:, -1, :]
    basket = ST @ weights
    return np.maximum(basket - strike, 0)


class Payoff:
    def __init__(self, payoff_function):
        self.payoff_function = payoff_function

    def evaluate(self, paths):
        return self.payoff_function(paths)
