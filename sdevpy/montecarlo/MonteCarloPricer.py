import numpy as np


class MonteCarloPricer:
    def __init__(self, df):
        self.df = df

    def price(self, paths, payoff):
        payoffs = payoff.evaluate(paths)
        return self.df * np.mean(payoffs)
