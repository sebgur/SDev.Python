""" Just to test things """
# import tensorflow as tf
import numpy as np
# import scipy.stats as sp
# import json
import matplotlib.pyplot as plt
# import requests, zipfile
# import io
#import pandas as pd
# from io import BytesIO
from sdevpy.volsurfacegen import sabrgenerator
from sdevpy.maths import metrics
from sdevpy.maths import optimization
from sdevpy.analytics import sabr

# ###################### Fit SABR straddles #######################################################
params_gen = {'LnVol': 0.25, 'Beta': 0.50, 'Nu': 0.50, 'Rho': -0.30}
fwd = 3.5 / 100
print(f"Generating params: {params_gen}")

# Calculate prices
expiries = np.asarray([1 / 52, 1 / 12, 0.5, 1.0, 5.0, 10.0])
print("Expiries ", expiries.shape, "\n", expiries)
n_expiries = expiries.shape[0]
spreads = np.asarray([-200, -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100, 200]).reshape(1, -1)
n_strikes = spreads.shape[1]
print("Spreads ", spreads.shape, "\n", spreads)
spreads = np.tile(spreads, (n_expiries, 1))
strikes = fwd + np.asarray(spreads) / 10000.0
SHIFT = 0.03
generator = sabrgenerator.SabrGenerator(SHIFT)
prices = generator.price_straddles_ref(expiries, strikes, fwd, params_gen)
weights = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# Objective function
def f(x, *args):
    params_ = {'LnVol': x[0], 'Beta': args[1], 'Nu': x[1], 'Rho': x[2]}
    # params_ = {'LnVol': x[0], 'Beta': x[1], 'Nu': x[2], 'Rho': x[3]}
    idx_ = args[0]
    generator_ = sabrgenerator.SabrGenerator(SHIFT)
    expiries_ = np.asarray([expiries[idx_]])
    strikes_ = np.asarray([strikes[idx_]])
    prices_ = generator_.price_straddles_ref(expiries_, strikes_, fwd, params_)
    return 10000.0 * metrics.rmsew(prices_, [prices[idx_]], [weights])


# Choose method
# method = 'Nelder-Mead'
# method = "Powell"
method = "L-BFGS-B"
# method = "DE"

# Create the optimizer
atol = 1e-4 # Relevant for DE
popsize = 5 # Relevant for DE
strategy = 'rand2exp' # Relevant for DE: best1bin, best1exp, best2exp, rand1exp
recombination = 0.7 # Relevant for DE
mutation = (0.5, 1.0) # Relevant for DE
# tol = 1e-8 # Relevant for non-DE
# optimizer = optimization.create_optimizer(method, atol=atol, popsize=popsize, strategy=strategy,
#                                           mutation=mutation, recombination=recombination,
#                                           tol=tol)

optimizer = optimization.MultiOptimizer(["L-BFGS-B", "DE"], mtol=atol, atol=atol, popsize=popsize, strategy=strategy,
                                          mutation=mutation, recombination=recombination)


# Define bounds and initial point (3d)
lw_bounds = [0.0001, 0.01, -0.9]
up_bounds = [10.0, 10.0, 0.9]
bounds = optimization.create_bounds(lw_bounds, up_bounds)
init_point = [0.20, 0.20, 0.0]

# # Define bounds and initial point (4d)
# lw_bounds = [0.0001, 0.1, 0.01, -0.9]
# up_bounds = [10.0, 0.9, 10.0, 0.9]
# bounds = optimization.create_bounds(lw_bounds, up_bounds)
# init_point = [0.20, 0.5, 0.20, 0.0]

import scipy.optimize as opt

# Optimize
fixed_beta = 0.50
params_opt = []
x = None
for i in range(n_expiries):
    print(f"Optimizing at T = {expiries[i]}...")
    result = optimizer.minimize(f, x0=init_point, args=(i, fixed_beta), bounds=bounds)
    # result = opt.differential_evolution(f, args=(i, fixed_beta), bounds=bounds, atol=1e-4,
    #                                     popsize=5, strategy='best2bin')
    # else:
    #     result = opt.differential_evolution(f, x0=x, args=(i, fixed_beta), bounds=bounds, atol=1e-5)

    x = result.x
    fun = result.fun
    params_opt.append({'LnVol': x[0], 'Beta': fixed_beta, 'Nu': x[1], 'Rho': x[2]})
    print(f'Optimum parameters at T = {expiries[i]:.3f}: {params_opt[i]}')
    print(f'Optimum objective at T = {expiries[i]:.3f}: {fun}')

# for key in result.keys():
#     if key in result:
#         print(key + "\n", result[key])

# Calculate implied vols
# opt_params = {'LnVol': x[0], 'Beta': x[1], 'Nu': x[2], 'Rho': x[3]}
target_ivs = []
optimum_ivs = []
shifted_fwd = fwd + SHIFT
plt_spreads = np.linspace(-300, 300, 100)
shifted_strikes = shifted_fwd + np.asarray(plt_spreads) / 10000.0
for i, expiry in enumerate(expiries):
    target_ivs_ = []
    optimum_ivs_ = []
    for sk in shifted_strikes:
        target_iv = sabr.implied_vol_vec(expiry, sk, shifted_fwd, params_gen)
        optimum_iv = sabr.implied_vol_vec(expiry, sk, shifted_fwd, params_opt[i])
        target_ivs_.append(target_iv)
        optimum_ivs_.append(optimum_iv)

    target_ivs.append(target_ivs_)
    optimum_ivs.append(optimum_ivs_)

# Plot
fig, axs = plt.subplots(3, 2, layout="constrained")
fig.suptitle("SABR Smiles Fit vs Target", size='x-large', weight='bold')
fig.set_size_inches(12, 8)

# PV
axs[0, 0].plot(plt_spreads, target_ivs[0], color='red', label='Target')
axs[0, 0].plot(plt_spreads, optimum_ivs[0], color='blue', label='Fit')
axs[0, 0].set_xlabel('Spread')
axs[0, 0].set_title(f"Fit vs Target at T={expiries[0]}")
axs[0, 0].legend(loc='upper right')

axs[0, 1].plot(plt_spreads, target_ivs[1], color='red', label='Target')
axs[0, 1].plot(plt_spreads, optimum_ivs[1], color='blue', label='Fit')
axs[0, 1].set_xlabel('Spread')
axs[0, 1].set_title(f"Fit vs Target at T={expiries[1]}")
axs[0, 1].legend(loc='upper right')

axs[1, 0].plot(plt_spreads, target_ivs[2], color='red', label='Target')
axs[1, 0].plot(plt_spreads, optimum_ivs[2], color='blue', label='Fit')
axs[1, 0].set_xlabel('Spread')
axs[1, 0].set_title(f"Fit vs Target at T={expiries[2]}")
axs[1, 0].legend(loc='upper right')

axs[1, 1].plot(plt_spreads, target_ivs[3], color='red', label='Target')
axs[1, 1].plot(plt_spreads, optimum_ivs[3], color='blue', label='Fit')
axs[1, 1].set_xlabel('Spread')
axs[1, 1].set_title(f"Fit vs Target at T={expiries[3]}")
axs[1, 1].legend(loc='upper right')

axs[2, 0].plot(plt_spreads, target_ivs[4], color='red', label='Target')
axs[2, 0].plot(plt_spreads, optimum_ivs[4], color='blue', label='Fit')
axs[2, 0].set_xlabel('Spread')
axs[2, 0].set_title(f"Fit vs Target at T={expiries[4]}")
axs[2, 0].legend(loc='upper right')

axs[2, 1].plot(plt_spreads, target_ivs[5], color='red', label='Target')
axs[2, 1].plot(plt_spreads, optimum_ivs[5], color='blue', label='Fit')
axs[2, 1].set_xlabel('Spread')
axs[2, 1].set_title(f"Fit vs Target at T={expiries[5]}")
axs[2, 1].legend(loc='upper right')

plt.show()

# ###################### column_stack #############################################################
# a = np.asarray(['a', 'b', 'c', 'd'])
# b = np.asarray(['1', '2', '3', '4'])
# c = np.asarray(['aa', 'bb', 'cc', 'dd'])
# d = np.asarray(['11', '22', '33', '44'])

# print(a.shape)
# e = np.column_stack((a, b))
# f = np.column_stack((c, d))
# print(e)
# print(f)
# print(np.column_stack((e, f)))

# URL = 'https://drive.google.com/file/d/10dKi82fW2arlKnOahNv9i5igfiydwMnc/view?usp=sharing'


# def download_file_from_google_drive(id, destination):
#     # URL = "https://docs.google.com/uc?export=download"

#     session = requests.Session()

#     print(URL)
#     print(id)
#     response = session.get(URL, params = { 'id' : id }, stream = True)
#     print(response)
#     token = get_confirm_token(response)
#     print(token)

#     if token:
#         params = { 'id' : id, 'confirm' : token }
#         response = session.get(URL, params = params, stream = True)

#     save_response_content(response, destination)    

# def get_confirm_token(response):
#     for key, value in response.cookies.items():
#         if key.startswith('download_warning'):
#             return value

#     return None

# def save_response_content(response, destination):
#     CHUNK_SIZE = 32768

#     with open(destination, "wb") as f:
#         for chunk in response.iter_content(CHUNK_SIZE):
#             if chunk: # filter out keep-alive new chunks
#                 f.write(chunk)

# id = '10dKi82fW2arlKnOahNv9i5igfiydwMnc'

# destination_ = ''

# download_file_from_google_drive(id, destination_)

# sample_url = 'https://1drv.ms/u/s!AivreF7B9rL4kK8Hx1vT4PRjtbE1iA?e=fxM7Kx' # OneDrive

# req = requests.get(sample_url)

# # Extract
# zipfile = zipfile.ZipFile(BytesIO(req.content))
# zipfile.extractall('New Folder')

# url = 'https://github.com/sebgur/SDev.Python/raw/main/models/stovol/SABR.zip'

# MODEL_NAME = 'SABR'
# OUPUT_ROOT = r'C:\temp\sdevpy\stovol\models'

# base_url = 'https://github.com/sebgur/SDev.Python/raw/main/models/stovol/'
# model_url = base_url + MODEL_NAME + ".zip"

# req = requests.get(model_url)

# filename = url.split('/')[-1]
# print("Downloading: " + filename)

# # Download
# with open(filename,'wb') as output_file:
#     output_file.write(req.content)
# print('Downloading Completed')

# # Extract
# zipfile = zipfile.ZipFile(BytesIO(req.content))
# zipfile.extractall(OUPUT_ROOT)


# url = "https://raw.githubusercontent.com/sebgur/SDev.Python/main/samples/McHeston_samples.tsv"

# https://github.com/sebgur/SDev.Python/blob/main/models/stovol/SABR/config.json
# https://raw.githubusercontent.com/sebgur/SDev.Python/main/models/stovol/SABR/config.json

# https://github.com/sebgur/SDev.Python/tree/main/models/stovol
# https://raw.githubusercontent.com/sebgur/SDev.Python/main/models/stovol

# download = requests.get(url).content

# df = pd.read_csv(io.StringIO(download.decode('utf-8')), sep='\t')
# print(df.head())

# parameters = { 'LnVol2': 0.2, 'Beta': 0.5}

# # print(parameters['LnVol'])

# if 'LnVol' in parameters:
#     print("found it")
# else:
#     print("boooo")


# strikes = [1, 2, 3, 4, 5]
# fwd = 2.5
# are_calls = [False if s < fwd else True for s in strikes]
# print(are_calls) 

# num_expiries = 2
# num_strikes = 3

# are_calls = [True, False, True]# * num_strikes
# are_calls = [are_calls] * num_expiries
# print(are_calls)
# are_calls = np.asarray(are_calls)
# print(are_calls.shape)
# print(are_calls.reshape(-1))

# is_call = [True, True, True]
# is_put = [False, False, False]
# is_opt = is_call + is_put
# print(is_opt)

# def f(t):
#     s1 = np.cos(2*np.pi*t)
#     e1 = np.exp(-t)
#     return s1 * e1

# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
# t3 = np.arange(0.0, 2.0, 0.01)

# num_rows = 2
# num_cols = 3
# title_ = "my big title"
# num_charts = num_rows * num_cols
# strikes = [0, 1, 2, 3, 4] * num_charts
# ref_disp = [0, 1, 2, 3, 4] * num_charts
# mod_disp = [0, 2, 4, 6, 8] * num_charts
# strikes = np.asarray(strikes).reshape(num_charts, 5)
# ref_disp = np.asarray(ref_disp).reshape(num_charts, 5)
# mod_disp = np.asarray(mod_disp).reshape(num_charts, 5)
# ylabel = "this is y"

# fig, axs = plt.subplots(num_rows, num_cols, layout="constrained")
# fig.suptitle(title_)
# for i in range(num_rows):
#     for j in range(num_cols):
#         k = num_cols * i + j
#         axs[i, j].plot(strikes[k], ref_disp[k], color='blue', label='Reference')

# plt.show()


# Json serialize/deserialize
# data = {
#     "user":
#       {
#           "name": "seb",
#           "age": 45,
#           "place": "Singapore"
#       }
# }

# file = r"C:\\temp\\sdevpy\\test.json"
# jsonstr = json.dumps(data)
# print(jsonstr)
# # with open(file, "w") as write:
# #     json.dump(data, write)

# newfiledata = open(file,)
# newdata = json.load(newfiledata)

# print(newdata['user']['age'])


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
