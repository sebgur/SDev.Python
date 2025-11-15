""" Simple example of binomial tree to value American options in Black-Scholes """
import numpy as np
import time
import matplotlib.pyplot as plt
from sdevpy.analytics import black
from sdevpy.tree import trees


def option_price(ttm, strike, is_call, is_american, spot, vol, rf_rate, div_rate, disc_rate,
                 method='trinomial', n_steps = 30):
    """ Price of an American option using binomial or trinomial trees under Black-Scholes model """
    payoff = Payoff(ttm, strike, is_call, is_american)
    return price(payoff, spot, vol, rf_rate, div_rate, disc_rate, method, n_steps)


def price(payoff, spot, vol, rf_rate, div_rate, disc_rate,
          method='trinomial', n_steps = 30):
    """ Price of a payoff using binomial or trinomial trees under Black-Scholes model """
    if method == 'binomial':
        tree = trees.BinomialTree(n_steps)
    else:
        tree = trees.TrinomialTree(n_steps)

    pv = tree.price(payoff, spot, vol, rf_rate, div_rate, disc_rate)

    return pv


if __name__ == "__main__":
    ttm = 2.5
    strike = 106.0
    is_call = False
    is_american = False
    vol = 0.20
    spot = 100.0
    rf_rate = 0.05
    div_rate = 0.02
    disc_rate = 0.03
    payoff = trees.Payoff(ttm, strike, is_call, is_american)

    # n_steps = 100
    repeat = 20
    bin_range = range(200, 1000, 20)
    tri_range = range(100, 500, 25)

    # Binomial
    bin_p = []
    bin_t = []
    for n_steps in bin_range:
        start = time.time()
        for _ in range(repeat):
            p = price(payoff, spot, vol, rf_rate, div_rate, disc_rate, 'binomial', n_steps)
        bin_t.append(time.time() - start)
        print(p)
        bin_p.append(p)

    # Trinomial
    tri_p = []
    tri_t = []
    for n_steps in tri_range:
        start = time.time()
        for _ in range(repeat):
            p = price(payoff, spot, vol, rf_rate, div_rate, disc_rate, 'trinomial', n_steps)
        tri_t.append(time.time() - start)
        print(p)
        tri_p.append(p)

    print(f"Binomial(price): {bin_p[-1]:.4f}")
    print(f"Binomial(time): {bin_t[-1]:.4f}")
    print(f"Trinomial(price): {tri_p[-1]:.4f}")
    print(f"Trinomial(time): {tri_t[-1]:.4f}")

    # Calculate vanilla price
    fwd = spot * np.exp((rf_rate - div_rate) * ttm)
    vanilla = np.exp(-disc_rate * ttm) * black.price(ttm, strike, is_call, fwd, vol)
    print(f"Vanilla CF: {vanilla:.4f}")
    cf_t = np.linspace(0, tri_t[-1], 50)
    cf_p = [vanilla] * 50

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(bin_t, bin_p, label='Binomial Tree Price')
    plt.plot(tri_t, tri_p, label='Trinomial Tree Price')
    plt.plot(cf_t, cf_p, label='Vanilla CF Price')
    # # plt.plot(steps_range, trinomial_prices, label='Trinomial Tree Price')
    # plt.hlines(bs_price, steps_range[0], steps_range[-1], colors='r', linestyles='dashed', label='Black-Scholes Price')
    plt.title('Convergence to Black-Scholes Price for Call Options')
    plt.xlabel('Runtime')
    plt.ylabel('Option Price')
    plt.legend()
    plt.show()


