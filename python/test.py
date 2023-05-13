""" Just to test things """
# import tensorflow as tf
import numpy as np
import scipy.stats as sp
# from maths.metrics import rmse
# import projects.xsabr_fit.sabrgenerator as sabr

from scipy.optimize import minimize_scalar
import py_vollib.black.implied_volatility as jaeckel

num_factors = 4
corr = np.zeros((num_factors, num_factors))
for i in range(num_factors):
    corr[i, i] = 1.0

print(corr)



time_steps = 2
factors = 3
sim = 5
row = ['a', 'b', 'c', 'd', 'e', 'f']
matrix = np.asarray([row] * sim)
print(matrix)
idx = 1
draws = [matrix[:,factors * idx:factors*(idx + 1)] for idx in range(time_steps)]
# draws = matrix.reshape(sim, time_steps * factors)
print(draws)
print(draws[0])
print(draws[1])
# print(draws[0:factors])

# N = sp.norm.cdf
# Ninv = sp.norm.ppf


# sampler = sp.qmc.Sobol(d=2, scramble=False)
# m=4
# uniforms = sampler.random_base2(m=m)
# print(uniforms)
# gaussians = Ninv(uniforms)
# print(gaussians)
# sample = sampler.random_base2(m=2)
# print(sample)
# print(2**m-1)

# def try_kwargs2(A="alpha", B=1):
#     return 0

# def try_kwargs(**kwargs):
#     kwargs.setdefault('A', "alpha")
#     kwargs.setdefault('B', "beta")

#     print(kwargs['A'])
#     print(kwargs['B'])

# try_kwargs(A="alpha", C="beta", B="gamma")
# try_kwargs2()

# spot = np.array([[10], [100], [1000]])
# print("spot\n", spot)

# print("spot shape ", spot.shape)

# exp_spot = np.expand_dims(spot, axis=1)
# print("Expanded spot\n", exp_spot)
# print("Expanded spot shape...", exp_spot.shape)

# strikes = np.asarray([1, 2, 3, 4]).reshape(1, -1, 1)
# print("Strikes shape...", strikes.shape)
# print("Strikes\n", strikes)

# payoff = exp_spot + strikes
# print(payoff)

# mean = np.mean(payoff, axis=0)
# print(mean)
