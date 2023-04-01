from products.Product import Product
import numpy as np


class Performance(Product):
    def __init__(self, expiry, strike, debug=False):
        self.debug = debug
        self.expiry = expiry
        self.strike = strike

    def disc_payoff(self, paths, disc_curve):
        shape = paths.shape
        num_samples = shape[0]
        num_underlyings = shape[1]
        payoff = np.ndarray((num_samples, 1))
        disc_rate = disc_curve  # This is just a placeholder. More generally we need a discount curve here
        df = np.exp(-self.expiry * disc_rate)
        for i in range(num_samples):
            # Calculate performances
            perf = np.ndarray(num_underlyings)
            for j_ in range(num_underlyings):
                perf[j_] = paths[i, j_]  # / fixings[j]
            if self.debug:
                print("Performances")
                print(perf)

            # Calculate basket performance (just the product for now)
            basket_perf = perf.prod(axis=0)

            # Calculate discounted payoff
            # payoff[i] = df_ * basket_perf
            payoff[i] = df * max(basket_perf - self.strike, 0)

        return payoff

    def closed_form_pv(self, disc_curve):
        return 0

    def has_closed_form(self):
        return True
