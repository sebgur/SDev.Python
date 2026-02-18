import numpy as np
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.CorrelationEngine import CorrelationEngine
from sdevpy.montecarlo.PathGenerator import PathGenerator
from sdevpy.montecarlo.Payoff import Payoff, basket_call_payoff
from sdevpy.montecarlo.MonteCarloPricer import MonteCarloPricer


#################### TODO #########################################################################
# * Call, Put, Straddle payoff, check against CF at constant vol
# * Extend to LV, use to check LV calib
# * Calculate vega through LV calib
# * Implement var swap spread payoff


if __name__ == "__main__":
    print("Hello")

    S0 = [100, 100]
    mu = [0.05, 0.05]
    sigma = [0.2, 0.3]

    corr = np.array([[1.0, 0.5],
                    [0.5, 1.0]])

    model = MultiAssetGBM(S0, mu, sigma)
    corr_engine = CorrelationEngine(corr)

    T = 1.0
    n_steps = 252
    n_paths = 10000

    generator = PathGenerator(model, corr_engine, T, n_steps)

    # Generate underlying paths?
    paths = generator.generate_paths(n_paths)

    weights = np.array([0.5, 0.5])
    strike = 100

    payoff = Payoff(lambda p: basket_call_payoff(p, weights, strike))

    pricer = MonteCarloPricer(df=0.98)

    price = pricer.price(paths, payoff)

    print("Price:", price)
