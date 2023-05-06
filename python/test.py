""" Just to test things """
# import tensorflow as tf
import numpy as np
# from maths.metrics import rmse
# import projects.xsabr_fit.sabrgenerator as sabr


spot = np.array([[10], [100], [1000]])
print("spot\n", spot)

print("spot shape ", spot.shape)

exp_spot = np.expand_dims(spot, axis=1)
print("Expanded spot\n", exp_spot)
print("Expanded spot shape...", exp_spot.shape)

strikes = np.asarray([1, 2, 3, 4]).reshape(1, -1, 1)
print("Strikes shape...", strikes.shape)
print("Strikes\n", strikes)

payoff = exp_spot + strikes
print(payoff)

mean = np.mean(payoff, axis=0)
print(mean)
