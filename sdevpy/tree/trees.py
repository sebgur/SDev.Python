""" Implementation of binomial and trinomial trees for Black-Scholes model """
import numpy as np
from abc import ABC, abstractmethod


class Tree(ABC):
    def __init__(self, n_steps):
        self.n_steps = n_steps

    def price(self, payoff, spot, vol, rf_rate, div_rate, disc_rate):
        ttm = payoff.maturity
        dt = ttm / self.n_steps
        df = np.exp(-disc_rate * dt)
        drift = rf_rate - div_rate

        # Calculate risk-neutral probabilities
        self.calculate_probabilities(dt, drift, vol)

        # Boundary at maturity
        spots = self.spot_vector(self.n_steps, spot)
        final = payoff.value(spots)

        # Backward iterations
        pv_vector = final
        for step in range(self.n_steps - 1, -1, -1):
            pv_vector = self.roll_back(step, payoff, pv_vector, spot, df)

        return pv_vector[0]

    @abstractmethod
    def spot_vector(self, step_idx, spot):
        pass

    @abstractmethod
    def calculate_probabilities(self, dt, drift, vol):
        pass

    @abstractmethod
    def roll_back(self, step_idx, payoff, pv_vector, spot, df):
        pass


class BinomialTree(Tree):
    def __init__(self, n_steps=30):
        super().__init__(n_steps)
        self.u = 0.0
        self.d = 0.0
        self.p = 0.0

    def spot_vector(self, step_idx, spot):
        u = self.u
        d = self.d
        return np.asarray([spot * u**i * d**(step_idx - i) for i in range(step_idx + 1)])

    def calculate_probabilities(self, dt, drift, vol):
        # Spot moves
        stdev = vol * np.sqrt(dt)
        self.u = np.exp(stdev)
        self.d = 1.0 / self.u

        # Transition probability
        self.p = (np.exp(drift * dt) - self.d) / (self.u - self.d)

    def roll_back(self, step_idx, payoff, pv_vector, spot, df):
        # Check size of old vector
        old_length = len(pv_vector)
        if old_length != step_idx + 2:
            raise RuntimeError("Incorrect vector size in binomial tree")

        # Roll-back to calculate continuation value
        p = self.p
        cont_v = np.asarray([df * (p * pv_vector[i + 1] + (1.0 - p) * pv_vector[i]) for i in range(step_idx + 1)])

        # Check american exercise
        if payoff.is_american:
            # Calculate exercise value
            spots = spot_vector(step_idx, spot)
            exer_v = payoff.exercise_value(spots)
            if exer_v.shape != cont_v.shape:
                raise RuntimeError("Incompatible shapes between continuation and exercise values")

            # Calculate optimum value
            pv = np.maximum(exer_v, cont_v) # ToDo: test
        else:
            pv = cont_v

        return pv


class TrinomialTree(Tree):
    def __init__(self, n_steps=30):
        super().__init__(n_steps)
        self.u = 0.0
        self.d = 0.0
        self.pu = 0.0
        self.pd = 0.0
        self.pm = 0.0

    def spot_vector(self, step_idx, spot):
        u = self.u
        d = self.d
        vec_u = np.cumprod(u * np.ones(step_idx))
        vec_d = np.cumprod(d * np.ones(step_idx))

        spots = np.concatenate((vec_d[::-1], [1.0], vec_u)) * spot
        return np.asarray(spots)

    def calculate_probabilities(self, dt, drift, vol):
        # Spot moves
        self.u = np.exp(vol * np.sqrt(2.0 * dt))
        self.d = 1.0 / self.u

        # Transition probabilities
        df_drift = np.exp(drift * dt / 2)
        df_vol_u = np.exp(vol * np.sqrt(dt / 2))
        df_vol_d = 1.0 / df_vol_u

        self.pu = ((df_drift - df_vol_d) / (df_vol_u - df_vol_d))** 2
        self.pd = ((df_vol_u - df_drift) / (df_vol_u - df_vol_d))** 2
        self.pm = 1.0 - self.pu - self.pd

    def roll_back(self, step_idx, payoff, pv_vector, spot, df):
        vec_stock = self.spot_vector(step_idx, spot)
        expectation = np.zeros(len(vec_stock))

        for j in range(len(expectation)):
            tmp = pv_vector[j] * self.pd
            tmp += pv_vector[j + 1] * self.pm
            tmp += pv_vector[j + 2] * self.pu

            expectation[j] = tmp

        return df * expectation


class Payoff:
    def __init__(self, ttm, strike, is_call, is_american):
        self.maturity = ttm
        self.strike = strike
        self.is_call = is_call
        self.is_american = is_american

    def value(self, spot):
        if self.is_call:
            return np.maximum(spot - self.strike, 0.0)
        else:
            return np.maximum(self.strike - spot, 0.0)

    def exercise_value(self, spot):
        return self.value(spot)
