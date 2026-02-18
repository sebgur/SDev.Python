import numpy as np
from sdevpy.montecarlo.FactorModel import MultiAssetGBM
from sdevpy.montecarlo.CorrelationEngine import CorrelationEngine
from sdevpy.montecarlo.PathGenerator import PathGenerator
from sdevpy.montecarlo.Payoff import Payoff, basket_call_payoff, callput_payoff, VanillaOption
from sdevpy.montecarlo.MonteCarloPricer import MonteCarloPricer
from sdevpy.analytics import black


#################### TODO #########################################################################
# * Call, Put, Straddle payoff, check against CF at constant vol
# * Extend to LV, use to check LV calib
# * Calculate vega through LV calib
# * Implement var swap spread payoff
# * Write a strict separation with the path generator, as the paths might come from
#   another engine (C++, C#, third-party, etc.)


if __name__ == "__main__":
    names = ['DAX', 'SPX', 'NKY']
    S0 = np.asarray([10, 100, 50])
    mu = np.asarray([0.02, 0.05, 0.04])
    sigma = np.asarray([0.2, 0.3, 0.1])
    n_assets = len(S0)
    df = 0.90

    corr = np.array([[1.0, 0.5, 0.1],
                    [0.5, 1.0, 0.1],
                    [0.1, 0.5, 1.0]])

    model = MultiAssetGBM(S0, mu, sigma)
    corr_engine = CorrelationEngine(corr)

    T = 1.0
    n_steps = 25
    n_paths = 100 * 1000

    generator = PathGenerator(model, corr_engine, T, n_steps)

    # Generate underlying paths: n_mc x (n_steps + 1) x n_assets
    paths = generator.generate_paths(n_paths)
    print(f"Number of assets: {n_assets}")
    print(f"Number of simulations: {n_paths}")
    print(f"Number of times points: {n_steps + 1}")
    print(f"Path shape: {paths.shape}")

    name = 'SPX'
    strike = 100
    optiontype = 'Call'

    # payoff = Payoff(lambda p: basket_call_payoff(p, strikes, weights))
    # payoff = Payoff(lambda p: callput_payoff(p, name, strike, optiontype, names))
    payoff = VanillaOption(name, strike, optiontype)
    payoff.set_pathindexes(names)

    mc = MonteCarloPricer(df=df)
    mc_price = mc.build(paths, payoff)

    name_idx = 1
    fwd = S0 * np.exp(mu * T)
    cf_price = df * black.price(T, strike, optiontype == 'Call', fwd[name_idx], sigma[name_idx])

    print("MC:", mc_price)
    print("CF:", cf_price)
