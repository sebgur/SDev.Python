""" Helper methods to compose a neural network model """
import os
import logging
import tensorflow as tf

kinits = tf.keras.initializers
klayers = tf.keras.layers


def add_hidden_layer(model, neurons, activation):
    """ Add hidden layer with Glorot initializer """
    init = kinits.glorot_normal
    model.add(klayers.Dense(neurons, activation=activation, kernel_initializer=init,
                            use_bias=True, bias_initializer=kinits.Constant(0.1)))


def compose_model(num_inputs, num_outputs, hidden_layers, neurons, dropout=0.2):
    """ Compose simple keras sequential model. The hidden layers are specified as a list of
        activation function names. """
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.Input(shape=(num_inputs,)))

    # Hidden layers
    for layer in hidden_layers:
        add_hidden_layer(model, neurons, layer)
        model.add(klayers.Dropout(dropout))

    # Output layer
    model.add(klayers.Dense(num_outputs))

    return model


# Turn off tensorflow warnings
def turn_off_ts_warnings():
    """ Turn off Tensorflow's warnings """
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
