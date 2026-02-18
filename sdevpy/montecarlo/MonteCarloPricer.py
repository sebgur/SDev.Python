import numpy as np


class MonteCarloPricer:
    def __init__(self, df):
        self.df = df

    def build(self, paths, payoff):
        payoffs = payoff.build(paths)
        return self.df * np.mean(payoffs)
