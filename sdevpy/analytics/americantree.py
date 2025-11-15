""" Simple example of binomial tree to value American options in Black-Scholes """
import numpy as np
from abc import ABC, abstractmethod
from sdevpy.analytics import black


def option_price(ttm, strike, is_call, is_american, spot, vol, rf_rate, div_rate, disc_rate,
                 method='trinomial', n_steps = 30):
    """ Price of an American option using binomial or trinomial trees under Black-Scholes model """
    payoff = Payoff(ttm, strike, is_call, is_american)
    return price(payoff, spot, vol, rf_rate, div_rate, disc_rate, method, n_steps)


def price(payoff, spot, vol, rf_rate, div_rate, disc_rate,
          method='trinomial', n_steps = 30):
    """ Price of a payoff using binomial or trinomial trees under Black-Scholes model """
    if method == 'binomial':
        return binomial_price(payoff, spot, vol, rf_rate, div_rate, disc_rate, n_steps)
    else:
        return trinomial_price(payoff, spot, vol, rf_rate, div_rate, disc_rate, n_steps)


class Tree(ABC):
    def __init__(self, n_steps=30):
        self.n_steps = n_steps

    def price(self, payoff, spot, vol, rf_rate, div_rate, disc_rate):
        ttm = payoff.maturity
        dt = ttm / n_steps
        df = np.exp(-disc_rate * dt)



def binomial_price(payoff, spot, vol, rf_rate, div_rate, disc_rate, n_steps = 4):
    """ Price of an American payoff using a binomial tree under Black-Scholes model """
    ttm = payoff.maturity
    dt = ttm / n_steps
    df = np.exp(-disc_rate * dt)

    stdev = vol * np.sqrt(dt)
    u = np.exp(stdev)
    d = 1 / u
    p = (np.exp((rf_rate - div_rate) * dt) - d) / (u - d)

    # Final boundary
    step = n_steps # Final step index
    spots = binomial_spot_vector(step, spot, u, d)
    final = payoff.value(spots)
    # final = final_boundary(step, payoff, spot, u, d)

    # Backward iterations
    pv_vector = final
    for step in range(n_steps - 1, -1, -1):
        pv_vector = roll_back_binomial(step, payoff, pv_vector, p, u, d, df)
        # print(f"Roll: {pv_vector}")

    return pv_vector[0]


def roll_back_binomial(step, payoff, old_pv, p, u, d, df):
    # Check size of old vector
    old_length = len(old_pv)
    if old_length != step + 2:
        raise RuntimeError("Incorrect vector size in binomial tree")

    cont_v = np.asarray([df * (p * old_pv[i + 1] + (1.0 - p) * old_pv[i]) for i in range(step + 1)])

    # Check american exercise
    if payoff.is_american:
        # Calculate exercise value
        spots = spot_vector(step, spot, u, d)
        exer_v = payoff.exercise_value(spots)
        if exer_v.shape != cont_v.shape:
            raise RuntimeError("Incompatible shapes between contination and exercise values")

        # Compare with continuation value
        pv = np.maximum(exer_v, cont_v) # ToDo: test
    else:
        pv = cont_v
    # new_vector = []

    return pv


def trinomial_price(payoff, spot, vol, rf_rate, div_rate, disc_rate, n_steps = 4):
    ttm = payoff.maturity
    dt = ttm / n_steps
    df = np.exp(-disc_rate * dt)
    strike = payoff.strike
    drift = rf_rate - div_rate
    is_call = payoff.is_call

    # Calculate risk-neutral probabilities
    pu = (
        (np.exp(drift * dt / 2) - np.exp(-vol * np.sqrt(dt / 2))) /
        (np.exp(vol * np.sqrt(dt / 2)) - np.exp(-vol * np.sqrt(dt / 2)))
    ) ** 2
    pd = (
        (np.exp(vol * np.sqrt(dt / 2)) - np.exp(drift * dt / 2)) /
        (np.exp(vol * np.sqrt(dt / 2)) - np.exp(-vol * np.sqrt(dt / 2)))
    ) ** 2
    pm = 1 - pu - pd

    # Final boundary
    spots = trinomial_spot_vector(n_steps, dt, spot, vol)
    final = payoff.value(spots)
    # final = np.maximum(s - strike, 0) if is_call else np.maximum(strike - s, 0)

    # Backward iterations
    pv_vector = final
    for i in range(1, n_steps + 1):
        vec_stock = trinomial_spot_vector(n_steps - i, dt, spot, vol)
        expectation = np.zeros(len(vec_stock))

        for j in range(len(expectation)):
            tmp = pv_vector[j] * pd
            tmp += pv_vector[j + 1] * pm
            tmp += pv_vector[j + 2] * pu

            expectation[j] = tmp
        # Discount option payoff
        pv_vector = df * expectation

    # Return the expected discounted value of the option at t=0
    return pv_vector[0]


def trinomial_spot_vector(steps, dt, spot, vol):
    up = np.exp(vol * np.sqrt(2 * dt))
    down = 1 / up

    vec_u = np.cumprod(up * np.ones(steps))
    vec_d = np.cumprod(down * np.ones(steps))

    stock_prices = np.concatenate((vec_d[::-1], [1.0], vec_u)) * spot
    return stock_prices


def binomial_spot_vector(step, spot, u, d):
    return np.asarray([spot * u**i * d**(step - i) for i in range(step + 1)])


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


if __name__ == "__main__":
    ttm = 1.0
    strike = 106.0
    is_call = False
    is_american = False
    vol = 0.20
    spot = 100.0
    rf_rate = 0.05
    div_rate = 0.05
    disc_rate = 0.03
    n_steps = 100
    payoff = Payoff(ttm, strike, is_call, is_american)
    binomial_p = price(payoff, spot, vol, rf_rate, div_rate, disc_rate, 'binomial', n_steps)
    print(f"Binomial: {binomial_p}")

    trinomial_p = price(payoff, spot, vol, rf_rate, div_rate, disc_rate, 'trinomial', n_steps)
    print(f"Trinomial: {trinomial_p}")

    # Calculate vanilla price
    fwd = spot * np.exp((rf_rate - div_rate) * ttm)
    # vanilla = fwd * np.exp(-disc_rate * ttm)
    vanilla = np.exp(-disc_rate * ttm) * black.price(ttm, strike, is_call, fwd, vol)
    print(f"Vanilla CF: {vanilla}")

