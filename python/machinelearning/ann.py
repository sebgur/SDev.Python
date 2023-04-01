import numpy as np
import keras.initializers as ki
from keras.layers import Dense, Dropout


def add_hidden_layer(model, nNeurons, nFeatures, activation):

    init = ki.RandomNormal(mean=0.0, stddev=1.0 / np.sqrt(nNeurons), seed=42)  # Initializer
    model.add(Dense(nNeurons, activation=activation, input_shape=(nFeatures,), kernel_initializer=init,
                    use_bias=True, bias_initializer=ki.Constant(0.1)))
    # model.add(Dropout(0.1))


def add_output_layer(model, n_inputs, n_outputs):

    init = ki.RandomNormal(mean=0.0, stddev=1.0 / np.sqrt(n_inputs), seed=42)  # Initializer
    model.add(Dense(n_outputs, activation='linear', kernel_initializer=init, use_bias=True, bias_initializer='zeros'))
