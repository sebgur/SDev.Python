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
    model.add(tf.keras.Input(num_inputs))

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


# Initialize Keras for reproducible results
# def init_keras():
#     """ Initialize Keras for reproducible results """
#     np.random.seed(42)
#     rn.seed(21)
#     session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
#                                               inter_op_parallelism_threads=1)
#     from tensorflow.keras import backend as kback
#     tf.random.set_seed(2019)
#     session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_config)
#     kback.set_session(session)


# def add_hidden_layer_old(model, neurons, features, activation):
#     init = kinits.RandomNormal(mean=0.0, stddev=1.0 / np.sqrt(neurons), seed=42)
#     model.add(klayers.Dense(neurons, activation=activation, input_shape=(features,),
#               kernel_initializer=init, use_bias=True, bias_initializer=kinits.Constant(0.1)))


# def add_output_layer(model, n_inputs, n_outputs):
#     init = ki.RandomNormal(mean=0.0, stddev=1.0 / np.sqrt(n_inputs), seed=42)
#     model.add(Dense(n_outputs, activation='linear', kernel_initializer=init, use_bias=True,
#     bias_initializer='zeros'))
