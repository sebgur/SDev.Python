import time
import tools.utils as utils
import tools.ann as ann
import tools.sabr as sabr
import tools.mlearning as mlearning
from keras.models import Sequential
from keras import optimizers
# from keras.models import load_model
import sklearn.utils as skutils


# Set ANN's topology
def set_sabr_model(n_features):
    model = Sequential()

    n_neurons = 64
    # Hidden layers
    ann.add_hidden_layer(model, n_neurons, n_features, 'relu')
    ann.add_hidden_layer(model, n_neurons, n_features, 'softmax')
    ann.add_hidden_layer(model, n_neurons, n_features, 'relu')
    # Output layer
    ann.add_output_layer(model, n_neurons, n_outputs)

    return model


# Train model on SABR data (csv)
def train_sabr_model(out_model_file, model=None):

    start = time.time()
    mlearning.init_keras()
    
    # Reuse model or specify new model
    if model is None:
        model = set_sabr_model(nFeatures)
        
    # Optimizer
    optimizer = optimizers.Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay,
                                amsgrad=False)
    
    # Fit
    model.compile(loss='mse', optimizer=optimizer)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize, shuffle=True, verbose=1)
    model.save(out_model_file)
    
    # Check MSEs
    check_train = model.predict(x_train)
    print('RMSE training: %.2f' % utils.rmse(check_train, y_train) + 'bps')
    check_test = model.predict(x_test)
    print('RMSE testing: %.2f' % utils.rmse(check_test, y_test) + 'bps')
    
    runtime = time.time() - start
    print('Runtime: %.0f' % runtime + 's')
    print('Training completed.')


# Specify training parameters
model_name = 'Hagan_SABR'
sample_file = 'outputs/' + model_name + '_samples.csv'
scaler_file = 'outputs/' + model_name + '_scaler.h5'
model_file = 'outputs/' + model_name + '_model.h5'
cleanse_data = False
shuffle_train = True
nFeatures = 7
n_outputs = 1
train_percent = 0.80  # 0.98
test_percent = 0.01  # 0.01
epochs = 10
batchSize = 10
learningRate = 0.001
decay = 1e-8

if cleanse_data:
    print("Cleansing data...")
    sabr.cleanse_sabr(sample_file)

print("Preparing data sets...")
x_train, x_test, x_val, y_train, y_test, y_val = mlearning.prepare_sets(train_percent, test_percent, nFeatures,
                                                                        sample_file, scaler_file)
if shuffle_train:
    print("Reshuffling training set...")
    x_train, y_train = skutils.shuffle(x_train, y_train, random_state=0)

init_model = None  # load_model(model_file) # Initialize to trained model to re-train (not working)
print("Training model...")
train_sabr_model(model_file, init_model)
sabr.view_sabr_smiles(model_file, scaler_file)
