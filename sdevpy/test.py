import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# import time

# a = np.asarray([[1, 2, 3], [4, 5, 6]])

# b = a[:, [1,2]]
# print(b)

a = np.asarray([1, 2, 3, 4])
print(a)
k = np.asarray([True, False, True, False])
a[k] = 0
print(a)

# a = [1, 2]
# b = [[1, 2], [3, 4], [5, 6]]

# weights = np.asarray([0.5, 0.5])
# state = np.asarray([b, b, b, b])
# print(state.shape)
# print(weights.shape)

# x = state * weights
# y = state @ weights
# print(x.shape)
# print(y.shape)

# print(f"State: {state}")
# print(f"Weights: {weights}")
# print(f"x: {x}")
# print(f"y: {y}")


# fwd_reshaped = fwd.reshape(-1, 1)
# print(fwd_reshaped.shape)
# print(fwd_reshaped)

# print(strikes / fwd_reshaped)

# # Define your objective function (normal signature, no grad needed)
# def f(x):
#     x_ = x[0]
#     a = 1.0
#     b = 2.0
#     c = 3.2
#     prod = a * b * c
#     bi = a * b + b * c + a * c
#     sum = a + b + c
#     value = 0.25 * np.power(x_, 4) - sum / 3.0 * np.power(x_, 3) + 0.5 * bi * x_**2 - prod * x_ + 1
#     # print(value)
#     return value


# # Define the search space with bounds
# space = [ Real(0.0, 5.0, name='x0') ]

# # Run Bayesian optimization
# start = time.time()
# result = gp_minimize(func=f, dimensions=space,            # search space with bounds
#     n_calls=40,                  # total number of evaluations (this IS respected!)
#     n_initial_points=2,          # number of random initial points before GP
#     acq_func='EI',              # acquisition function: 'EI', 'LCB', or 'PI'
#     random_state=42,            # for reproducibility
#     verbose=False                # print progress
# )
# end = time.time()
# print(f"Time: {end - start}s")
# print(f"\nOptimization complete!")
# print(f"Best x: {result.x}")
# print(f"Best f(x): {result.fun}")
# print(f"Total evaluations: {len(result.func_vals)}")

# # Access the optimization history
# print(f"\nAll evaluated points:")
# # for i, (point, value) in enumerate(zip(result.x_iters, result.func_vals)):
# #     print(f"  {i+1}. x = {point}, f(x) = {value}")
