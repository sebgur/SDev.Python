""" Simple example of binomial tree to value American options in Black-Scholes """
import numpy as np
from sdevpy.analytics import black


def price(ttm, strike, is_call, is_american, spot, vol, rf_rate, div_rate, disc_rate, n_steps = 30):
    """ Price of an American option using a binomial tree under Black-Scholes model """
    payoff = Payoff(ttm, strike, is_call)
    return price(payoff, spot, vol, rf_rate, div_rate, disc_rate)


def price(payoff, spot, vol, rf_rate, div_rate, disc_rate, n_steps = 4):
    """ Price of an American payoff using a binomial tree under Black-Scholes model """
    ttm = payoff.maturity
    dt = ttm / n_steps
    # print(dt)
    # print(vol**2 / (rf_rate - div_rate))
    stdev = vol * np.sqrt(dt)
    u = np.exp(2.0 * stdev)
    d = np.exp(-2.0 * stdev)
    p = (np.exp((rf_rate - div_rate) * dt) - d) / (u - d)
    df = np.exp(-disc_rate * dt)

    # Final boundary
    step = n_steps # Final step index
    final = final_boundary(step, payoff, spot, u, d)
    # print(f"Final: {final}")

    # Backward iteration
    pv_vector = final
    for step in range(n_steps - 1, -1, -1):
        pv_vector = roll_back(step, payoff, pv_vector, p, u, d, df)
        # print(f"Roll: {pv_vector}")

    return pv_vector[0]


def final_boundary(step, payoff, spot, u, d):
    if step < 1:
        raise RuntimeError("Binomial trees require at least 1 step")

    spots = spot_vector(step, spot, u, d)
    return payoff.value(spots)


def roll_back(step, payoff, old_pv, p, u, d, df):
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
        pv = np.maximum(exer_v, cont_v)
    else:
        pv = cont_v
    # new_vector = []

    return pv


def spot_vector(step, spot, u, d):
    return np.asarray([spot * u**(step - i) * d**i for i in range(step + 1)])


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
    ttm = 1.5
    strike = 105.0
    is_call = True
    is_american = False
    vol = 0.10
    spot = 100.0
    rf_rate = 0.0
    div_rate = 0.00
    disc_rate = 0.0
    n_steps = 30
    payoff = Payoff(ttm, strike, is_call, is_american)
    p = price(payoff, spot, vol, rf_rate, div_rate, disc_rate, n_steps)
    print(f"Tree: {p}")

    # Calculate vanilla price
    fwd = spot * np.exp((rf_rate - div_rate) * ttm)
    vanilla = np.exp(-disc_rate * ttm) * black.price(ttm, strike, is_call, fwd, vol)
    print(f"Vanilla CF: {vanilla}")

