import numpy as np
from tools.settings import apply_settings
from tools.black import black_performance
from tools.timer import Stopwatch
from tools.LearningModel import LearningModel, read_learning_model
from montecarlo.Simulation import mc_simulation
import tools.mlearning as mlearning
import tools.ann as ann
from tools.utils import rmse
from CfCallback import CfCallback
from keras.models import Sequential
from keras import optimizers
# from keras import backend as back_end
# import tensorflow as tf
import matplotlib.pyplot as plt
from montecarlo.BlackSimulator import BlackSimulator
from products.Performance import Performance
from enum import Enum
import random

apply_settings()

# ############################ Notes ######################################################################
# * Implement MC in tensorflow and get its Greeks by AAD
# * Implement CF delta. Try to do it by AAD and compare. Then do vega by AAD.
# * Make methods that return PV and Greeks in one go in BlackFormula, or BlackPrice that
#   returns only the PV. Do the same thing for MC?
# * Define an initialization method that simply "adds" scenarios in the sense of adding a bunch of
#     identical initialization points at one specific place in the initial space.
# * To compare runtime between MC and DL: do it on multiple estimations, not using Greeks as Greeks could
#     be done by AAD in MC. Select a bunch of points and compare how it goes  with the number of points
# * Make the discount rate an initial parameter to get the DV01?


# ############################ Enums ######################################################################
class InitType(Enum):
    Method1 = 1
    Method2 = 2
    Method3 = 3


# ############################ Runtime configuration ######################################################
print("<><><><> Setting up runtime configuration <><><><><><><><><><><><>")
debug = False
train = False
save_trained = False
display_fit = False
display_history = False
do_mc = False
output_path = ""
root_file_name = "fk_model"

# Model and payoff parameters
spot = 100.0
strike = 100.0
expiry = 1.25
repo_rate = 0.01
disc_rate = 0.015
div_rate = 0.005
vol = 0.20
sqrt_t = np.sqrt(expiry)
df = np.exp(-disc_rate * expiry)
num_underlyings = 1
fixings = [85]
# fixings = [85, 92]

# Monte-Carlo parameters
num_mc = 1000000
rng_seed_mc = 42
rng_mc = np.random.RandomState(rng_seed_mc)

# ## Deep-Learning parameters ##
num_samples = 1 * 1000 * 1000
init_type = InitType['Method2']
num_iterations = 2
regenerate_init = True
epochs = 10
batch_size = 10 * 1000
num_neurons = 8  # default 32
hidden_layers = ['softmax', 'relu', 'softmax']
loss_function = 'mse'  # default 'mse'
num_features = 2 * num_underlyings  # Learning on spots and vols
num_output = 1
lr = 0.001  # default 0.001
decay = 1e-8  # default 1e-8
eps = 1e-7  # default 1e-7

rng_seed_dl = 42
rng_dl = np.random.RandomState(rng_seed_dl)

# Initialization range
spot_min = spot - 10
spot_max = spot + 10
vol_min = vol - 1e-8
vol_max = vol + 1e-8

# ## Path simulator and payoff objects ##
print("Setting up path simulator...")
path_simulator = BlackSimulator(repo_rate, div_rate, expiry, debug)

print("Setting up payoff...")
product = Performance(expiry, strike, debug)
has_closed_form = product.has_closed_form()

# ## Display settings ##
print("Setting up display configuration...")
num_disp = 100
disp_vol = vol
disp_min = 90
disp_max = 110
disp_spot = np.linspace(disp_min * 1.0, disp_max * 1.0, num_disp)
x_disp = np.ndarray((num_disp, num_features))
for disp in range(num_disp):
    for u in range(num_underlyings):
        x_disp[disp, 2 * u] = disp_spot[disp]
        x_disp[disp, 2 * u + 1] = disp_vol

if debug:
    print(x_disp)

# ## Validation and checks ##
# Setting up comparison ranges

# ############################ Closed-Form method #########################################################
y_ref = np.ndarray((num_disp, 1))
if has_closed_form:
    print("Calculating closed-form...")
    df = np.exp(-disc_rate * expiry)
    cf_spot_vol = np.ndarray(shape=(2, 1))
    cf_spot_vol[0, 0] = spot
    cf_spot_vol[1, 0] = vol
    cf_pv = df * black_performance(cf_spot_vol, repo_rate, div_rate, expiry, strike, fixings)

    for disp in range(num_disp):
        y_ref[disp] = df * black_performance(x_disp[disp], repo_rate, div_rate, expiry, strike, fixings)

    if debug:
        print(y_ref)

else:
    cf_pv, cf_delta, cf_gamma, cf_vega = None, None, None, None


# ############################ Monte-Carlo method #########################################################
mc_timer = Stopwatch("MC")
mc_timer.trigger()
if do_mc:
    print("<><><><> Monte-Carlo method <><><><><><><><><><><><><><><><>")
    mc_pv, mc_delta, mc_gamma, mc_vega = mc_simulation(num_underlyings, path_simulator, product, disc_rate,
                                                       spot, vol, num_mc, rng_mc)
else:
    mc_pv, mc_delta, mc_gamma, mc_vega = None, None, None, None

mc_timer.stop()

# ############################ Deep-learning method #######################################################
print("<><><><> Deep-Learning method <><><><><><><><><><><><><><><><><><>")
dl_timer = Stopwatch("DL")
dl_timer.trigger()
dl_init_timer = Stopwatch("DL-gen")
dl_train_timer = Stopwatch("DL-train")
dl_path_timer = Stopwatch("DL-path")

if train:
    # ############################ Composing learning model ###############################################
    print("<><> Composing learning model <><>")
    print("Neurons per layer: " + str(num_neurons))
    print("Hidden layers: " + str(hidden_layers))
    print("Optimizer: " + "Adam")

    mlearning.init_keras()
    ml_model = Sequential()

    # Hidden layers
    for hl in hidden_layers:
        ann.add_hidden_layer(ml_model, num_neurons, num_features, hl)

    # Output layer
    ann.add_output_layer(ml_model, num_neurons, num_output)

    # Optimizer
    optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=eps, decay=decay,
                                amsgrad=False)

    ml_model.compile(loss=loss_function, optimizer=optimizer)
    # ml_model.compile(loss='mse', optimizer=optimizer)
    learning_model = LearningModel(ml_model)

    # ############################ Learning ################################################################
    call_back = CfCallback(x_disp, y_ref)  # To keep track of progress
    x_set = y_set = None
    for dl_iter in range(num_iterations):
        print("<><> Learning iteration " + str(dl_iter + 1) + "/" + str(num_iterations) + " <><>")

        if regenerate_init or dl_iter == 0:
            # Iterate
            dl_init_timer.trigger()

            # Generate initial spots
            print("Generating initial spots...")
            if init_type == InitType.Method1:
                init_spot = rng_dl.uniform(spot_min, spot_max, (num_samples, num_underlyings))
            elif init_type == InitType.Method2:
                init_spot = rng_dl.normal(spot, spot * vol, (num_samples, num_underlyings))
                for i in range(num_samples):
                    for j in range(num_underlyings):
                        if init_spot[i][j] < spot_min:
                            init_spot[i][j] = spot
                        elif init_spot[i][j] > spot_max:
                            init_spot[i][j] = spot
            elif init_type == InitType.Method3:
                spot_grid = np.linspace(spot_min, spot_max, 51)
                spot_grid[2] = 190
                spot_grid[3] = 190
                spot_grid[15] = 190
                spot_grid[47] = 190
                spot_grid[48] = 190
                print(spot_grid)

                spot_roots = random.choices(spot_grid, k=num_samples)
                if debug:
                    print(spot_roots)

                init_spot = np.ndarray(shape=(num_samples, num_underlyings))
                for i in range(num_samples):
                    for j in range(num_underlyings):
                        init_spot[i, j] = spot_roots[i]
            else:
                raise RuntimeError("Unknown initialization method")

            # Generate initial vols
            print("Generating initial vols...")
            init_vol = rng_dl.uniform(vol_min, vol_max, (num_samples, num_underlyings))

            if debug:
                print("Initial spots")
                print(init_spot)
                print("Initial vols")
                print(init_vol)

            dl_init_timer.stop()

            # #### Calculating paths ############
            dl_path_timer.trigger()

            # Calculating spot paths
            print("Calculating future spots...")
            future_spot = path_simulator.build_paths(init_spot, init_vol, rng_dl)
            if debug:
                print("Future spots")
                print(future_spot)

            # Calculating discounted payoff paths
            print("Calculating discounted payoffs...")
            disc_payoff = product.disc_payoff(future_spot, disc_rate)
            if debug:
                print("Discounted payoffs")
                print(disc_payoff)

            dl_path_timer.stop()

            # Gather simulation data into ML training format
            print("Gathering dataset...")
            x_set = np.ndarray((num_samples, num_features))
            for s in range(num_samples):
                for u in range(num_underlyings):
                    x_set[s, 2 * u] = init_spot[s, u]
                    x_set[s, 2 * u + 1] = init_vol[s, u]

            y_set = disc_payoff
            if debug:
                print("x-dataset")
                print(x_set)
                print("y-dataset")
                print(y_set)

        # ############################ Training ############################################################
        dl_train_timer.trigger()
        print("Training...")
        learning_model.train(x_set, y_set, epochs, batch_size, call_back)
        dl_train_timer.stop()

    # Retrieve convergence information
    hist_epochs, hist_loss, hist_acc = learning_model.convergence()

    if save_trained:
        print("Saving trained model to file root " + root_file_name)
        learning_model.save_to(output_path, root_file_name)
else:
    x_set = y_set = hist_epochs = hist_loss = hist_acc = None
    learning_model = read_learning_model(output_path, root_file_name)

dl_timer.stop()

# ############# Value with trained algorithm ###############################################################
print("<> Value with trained algorithm")
# Calculate PV and Greeks with model
print("Calculate PV and Greeks...")
md_x = np.ndarray(shape=(1, 2))
md_x[0, 0] = spot
md_x[0, 1] = vol
md_pv, diff = learning_model.calculate(md_x, diff=True)
# print(md_pv)
md_delta = diff[:, 0]
md_vega = diff[:, 1]
# print(md_delta)
# print(md_vega)

# Calculate display set with model
print("Calculate scenarios...")
y_mod = learning_model.predict(x_disp)


# ############################ Displaying result ###########################################################
print("<><><><> Plotting results <><><><><><><><><><><><><><><><><><><><>")
if display_fit and train:
    print("Plotting model vs training data...")
    plt.title("Fit")
    plt.xlabel('Spot')
    plt.ylabel('PV')
    plt.plot(disp_spot, y_mod, color='red', label='Learning model')
    # plt.plot(disp_spot, y_ref, color='red', label='Closed-Form')
    x_scat = x_set[:, 0]
    plt.scatter(x_scat, y_set, color='blue', alpha=1.0 / 255, label='Full series')
    plt.legend(loc='upper right')
    plt.show()

if display_history and train:
    print("Plotting loss history...")
    plt.title("Loss history")
    plt.plot(hist_epochs, hist_loss)
    # plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if has_closed_form:
    print("Plotting results against closed-form...")
    plt.ioff()
    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(hspace=0.40)
    # Convergence to Closed-Form
    if train:
        plt.subplot(1, 2, 1)
        plt.title("Convergence to Closed-Form")
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.plot(hist_epochs, hist_acc, color='blue')

    # Model vs Closed-Form
    plt.subplot(1, 2, 2)
    rmse_ = rmse(y_ref, y_mod)
    plt.title("Model vs Closed-Form, rmse={a:,.0f}".format(a=rmse_))
    plt.xlabel('Spot')
    plt.ylabel('PV')
    plt.plot(disp_spot, y_ref, color='blue', label='Closed-Form')
    plt.plot(disp_spot, y_mod, color='red', label='NN')
    plt.legend(loc='upper right')
    # plt.show()

# ############################ Numerical checks ###########################################################
print("<><><><> Numerical checks <><><><><><><><><><><><><><><><><><><><>")
# Check PV
if do_mc:
    print("MC PV: {a:,.4f}".format(a=mc_pv))
print("CF PV: {a:,.4f}".format(a=cf_pv[0]))
print("MD PV: {a:,.4f}".format(a=md_pv[0]))
print("MD Delta: {a:,.4f}".format(a=md_delta[0]))
print("MD Vega: {a:,.4f}".format(a=md_vega[0]))

# ############################ Timers #####################################################################
print("<><><><> Timers <><><><><><><><><><><><><><><><><><><><><><><><><>")
if do_mc:
    mc_timer.print()

if train:
    dl_init_timer.print()
    dl_path_timer.print()
    dl_train_timer.print()

dl_timer.print()
