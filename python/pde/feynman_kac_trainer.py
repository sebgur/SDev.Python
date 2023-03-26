#
#
# def train_feynman_kac():
#     # Generate initial spots
#     print("Generating initial spots...")
#     if init_type == InitType.Method1:
#         init_spot = rng_dl.uniform(spot_min, spot_max, (num_samples, num_underlyings))
#     elif init_type == InitType.Method2:
#         init_spot = rng_dl.normal(spot, spot * vol, (num_samples, num_underlyings))
#         for i in range(num_samples):
#             for j in range(num_underlyings):
#                 if init_spot[i][j] < spot_min:
#                     init_spot[i][j] = spot
#                 elif init_spot[i][j] > spot_max:
#                     init_spot[i][j] = spot
#     elif init_type == InitType.Method3:
#         spot_grid = np.linspace(spot_min, spot_max, 51)
#         spot_grid[2] = 190
#         spot_grid[3] = 190
#         spot_grid[15] = 190
#         spot_grid[47] = 190
#         spot_grid[48] = 190
#         print(spot_grid)
#
#         spot_roots = random.choices(spot_grid, k=num_samples)
#         if debug:
#             print(spot_roots)
#
#         init_spot = np.ndarray(shape=(num_samples, num_underlyings))
#         for i in range(num_samples):
#             for j in range(num_underlyings):
#                 init_spot[i, j] = spot_roots[i]
#     else:
#         raise RuntimeError("Unknown initialization method")
#
#     # Generate initial vols
#     print("Generating initial vols...")
#     init_vol = rng_dl.uniform(vol_min, vol_max, (num_samples, num_underlyings))
#
#     if debug:
#         print("Initial spots")
#         print(init_spot)
#         print("Initial vols")
#         print(init_vol)
#
#     dl_init_timer.stop()
#
#     # #### Calculating paths ############
#     dl_path_timer.trigger()
#
#     # Calculating spot paths
#     print("Calculating future spots...")
#     future_spot = path_simulator.build_paths(init_spot, init_vol, rng_dl)
#     if debug:
#         print("Future spots")
#         print(future_spot)
#
#     # Calculating discounted payoff paths
#     print("Calculating discounted payoffs...")
#     disc_payoff = product.disc_payoff(future_spot, disc_rate)
#     if debug:
#         print("Discounted payoffs")
#         print(disc_payoff)
#
#     dl_path_timer.stop()
#
#     # Gather simulation data into ML training format
#     print("Gathering dataset...")
#     x_set = np.ndarray((num_samples, num_features))
#     for s in range(num_samples):
#         for u in range(num_underlyings):
#             x_set[s, 2 * u] = init_spot[s, u]
#             x_set[s, 2 * u + 1] = init_vol[s, u]
#
#     y_set = disc_payoff
#     if debug:
#         print("x-dataset")
#         print(x_set)
#         print("y-dataset")
#         print(y_set)
#
#     dl_train_timer.trigger()
