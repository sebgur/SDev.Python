import numpy as np
# from skopt import gp_minimize
# from skopt.space import Real
# import time


x = np.array([1, 2, 3, 4, 5])
x_new = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
y = np.array([10, 20, 30, 40, 50])

# Left-continuous (takes the value of the next point)
# Also called "previous" or "left" step
indices = np.searchsorted(x, x_new, side='right')
print(indices)
print(np.clip(indices, 0, len(y) - 1))
y_left = y[np.clip(indices, 0, len(y) - 1)]
# print(y_left)
# [20, 30, 40, 50]

# Right-continuous (takes the value of the previous point)
# Also called "next" or "right" step
indices = np.searchsorted(x, x_new, side='left') - 1
print(indices)
print(np.clip(indices, 0, len(y) - 1))
y_right = y[np.clip(indices, 0, len(y) - 1)]
# print(y_right)
# [10, 20, 30, 40]


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
