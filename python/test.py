""" Just to test things """
# import tensorflow as tf
import numpy as np
import scipy.stats as sp
# from maths.metrics import rmse
# import projects.xsabr_fit.sabrgenerator as sabr


# Merge two data samples


# expiries = np.asarray([0.5, 2.5]).reshape(-1, 1)
# strikes = np.asarray([[111, 222, 333], [44, 55, 66]])
# num_expiries = 2
# num_strikes = 3
# num_points = num_expiries * num_strikes
# md_inputs = np.ones((num_points, 3))
# md_inputs[:, 0] = np.repeat(expiries, num_strikes)
# md_inputs[:, 1] = strikes.reshape(-1)
# for i in range(6):
#     md_inputs[i, 2] = i + 1
# print(md_inputs)

# vols = md_inputs[:, 2]
# print(vols)
# svols = vols.reshape(2, 3)
# print(svols)

# vec_a = ['a', 'b', 'c']
# vec_b = ['1', '2', '3']

# for (a, b) in zip(vec_a, vec_b):
#     print(a + b)


# expiries = np.asarray([1, 2, 3]).reshape(-1, 1)
# print(expiries)
# # mod_expiries = np.tile(expiries, (2, 1))
# mod_expiries = np.repeat(expiries, 2)
# print(mod_expiries)

# strikes = np.asarray([[1, -1], [2, -2], [3, -3]])
# print(strikes)
# mod_strikes = strikes.reshape(-1, 1)
# print(mod_strikes)

# a = np.asarray([[1, 2, 3], [4, 5, 6]])
# print(a)
# print(a.shape)
# b = np.concatenate((a, -a), axis=0)
# print(b)
# print(b.shape)


# from scipy.optimize import minimize_scalar
# import py_vollib.black.implied_volatility as jaeckel

# num_factors = 4
# corr = np.zeros((num_factors, num_factors))
# for i in range(num_factors):
#     corr[i, i] = 1.0

# print(corr)



# time_steps = 2
# factors = 3
# sim = 5
# row = ['a', 'b', 'c', 'd', 'e', 'f']
# matrix = np.asarray([row] * sim)
# print(matrix)
# idx = 1
# draws = [matrix[:,factors * idx:factors*(idx + 1)] for idx in range(time_steps)]
# # draws = matrix.reshape(sim, time_steps * factors)
# print(draws)
# print(draws[0])
# print(draws[1])
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
