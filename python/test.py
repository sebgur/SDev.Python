""" Just to test things """
import numpy as np
import projects.xsabr_fit.sabrgenerator as sabr


NUM_TEST = 4
spread_grid = np.linspace(-300, 300, num=NUM_TEST)

generator = sabr.SabrGenerator()

TEST_PARAMS = { 'TTM': 1.2, 'F': 0.035, 'LNVOL': 0.20, 'Beta': 0.5, 'Nu': 0.55, 'Rho': -0.25 }

prices = generator.price_strike_ladder(None, spread_grid, TEST_PARAMS)

print(prices)
