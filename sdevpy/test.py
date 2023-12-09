""" Just to test things """
# import tensorflow as tf
# import numpy as np
# import scipy.stats as sp
# import json
# import matplotlib.pyplot as plt
# import requests, zipfile
# import io
#import pandas as pd
# from io import BytesIO

# To know Python version is 32 or 64 bits
import struct
print(struct.calcsize("P") * 8)

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
